#!/usr/bin/env python3
"""QLoRA fine-tuning for CSS 360 syllabus instruction/response JSONL.

Designed for a single Tillicum GPU via Slurm. Supports a tiny --smoke-test mode
that still measures step timing and estimates full-run duration / GPU hours.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without GPU / model download)
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = ("instruction", "response")
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_SEED = 360
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class TrainingDataError(ValueError):
    """Raised when train/validation JSONL input is invalid."""


@dataclass(frozen=True)
class ResolvedRunConfig:
    mode: str
    model_id: str
    train_path: str
    validation_path: str
    output_dir: str
    max_seq_length: int
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    weight_decay: float
    seed: int
    gpu_count: int
    smoke_test: bool
    max_steps: int | None
    train_example_count: int
    validation_example_count: int
    full_train_example_count: int
    full_validation_example_count: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float


def load_instruction_response_jsonl(path: str | Path) -> list[dict[str, str]]:
    """Load JSONL records with non-blank instruction/response fields."""
    file_path = Path(path)
    if not file_path.is_file():
        raise TrainingDataError(f"Input file does not exist: {file_path}")

    records: list[dict[str, str]] = []
    text = file_path.read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise TrainingDataError(
                f"Malformed JSONL at line {line_number}: blank line is not allowed"
            )
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingDataError(
                f"Malformed JSONL at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise TrainingDataError(
                f"Malformed JSONL at line {line_number}: record must be a JSON object"
            )
        instruction = payload.get("instruction")
        response = payload.get("response")
        if not isinstance(instruction, str) or not instruction.strip():
            raise TrainingDataError(
                f"Blank instruction at line {line_number}"
            )
        if not isinstance(response, str) or not response.strip():
            raise TrainingDataError(
                f"Blank response at line {line_number}"
            )
        records.append(
            {
                "instruction": instruction,
                "response": response,
            }
        )
    return records


def effective_batch_size(
    *,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    gpu_count: int,
) -> int:
    if per_device_batch_size < 1:
        raise ValueError("per_device_batch_size must be >= 1")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    if gpu_count < 1:
        raise ValueError("gpu_count must be >= 1")
    return per_device_batch_size * gradient_accumulation_steps * gpu_count


def estimate_optimizer_steps(
    *,
    train_example_count: int,
    epochs: float,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    gpu_count: int,
) -> int:
    """Estimate full-run optimizer steps from dataset size and batching."""
    if train_example_count < 1:
        raise ValueError("train_example_count must be >= 1")
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    batch = effective_batch_size(
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gpu_count=gpu_count,
    )
    steps_per_epoch = max(1, math.ceil(train_example_count / batch))
    return int(math.ceil(steps_per_epoch * float(epochs)))


def resolve_max_steps(
    *,
    smoke_test: bool,
    smoke_max_steps: int | None,
    train_example_count: int,
    epochs: float,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    gpu_count: int,
) -> int:
    """Resolve the explicit ``max_steps`` value handed to SFTConfig.

    Smoke runs keep their fixed tiny cap. Full runs must pass an explicit
    optimizer-step budget because Hugging Face Trainer derives its own budget
    as ``floor(len(dataloader) / gradient_accumulation_steps) * epochs``. When
    the example count is not divisible by the effective batch size that floor
    silently drops the trailing partial accumulation group of every epoch, so
    the run stops materially short of ``num_train_epochs`` (e.g. 37 examples /
    accumulation 8 / 3 epochs stops at 12 steps, epoch ~2.43, instead of 15).
    The ceil-based estimator is the single source of truth for the budget.
    """
    if smoke_test:
        if smoke_max_steps is None or smoke_max_steps < 1:
            raise ValueError("smoke_max_steps must be >= 1 for smoke runs")
        return int(smoke_max_steps)
    return estimate_optimizer_steps(
        train_example_count=train_example_count,
        epochs=epochs,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gpu_count=gpu_count,
    )


def evaluate_training_length(
    *,
    completed_steps: int,
    intended_steps: int,
) -> dict[str, Any]:
    """Compare completed optimizer steps against the intended step budget."""
    if intended_steps < 1:
        raise ValueError("intended_steps must be >= 1")
    completed = max(0, int(completed_steps))
    missing = max(0, intended_steps - completed)
    return {
        "intendedOptimizerSteps": intended_steps,
        "completedSteps": completed,
        "missingOptimizerSteps": missing,
        "completedStepRatio": completed / float(intended_steps),
        "trainingLengthSatisfied": missing == 0,
    }


def average_seconds_per_step(
    step_durations_seconds: list[float],
    *,
    exclude_first: bool = False,
) -> float | None:
    """Return mean step duration, or None when no usable samples exist."""
    samples = list(step_durations_seconds)
    if exclude_first and len(samples) > 1:
        samples = samples[1:]
    if not samples:
        return None
    return sum(samples) / len(samples)


def choose_conservative_step_average(
    all_steps_avg: float | None,
    exclude_first_avg: float | None,
) -> float | None:
    """Prefer the larger (more conservative) positive average when both exist."""
    candidates = [value for value in (all_steps_avg, exclude_first_avg) if value and value > 0]
    if not candidates:
        return None
    return max(candidates)


def estimate_training_only_seconds(
    *,
    average_seconds_per_step_value: float | None,
    estimated_optimizer_steps: int,
) -> float | None:
    if average_seconds_per_step_value is None or average_seconds_per_step_value <= 0:
        return None
    if estimated_optimizer_steps < 1:
        return None
    return average_seconds_per_step_value * estimated_optimizer_steps


def estimate_conservative_total_seconds(
    *,
    estimated_training_only_seconds: float | None,
    model_load_seconds: float,
    evaluation_seconds: float,
    epochs: float,
    average_seconds_per_step_value: float | None,
) -> float | None:
    """Add model load, eval, and rough checkpoint overhead to training-only estimate."""
    if estimated_training_only_seconds is None:
        return None
    epoch_count = max(1.0, float(epochs))
    # Use measured eval time as a per-epoch proxy when available.
    eval_overhead = max(0.0, evaluation_seconds) * epoch_count
    # Rough checkpoint write cost: a few step-equivalents per epoch.
    step_proxy = average_seconds_per_step_value or 0.0
    checkpoint_overhead = max(0.0, step_proxy * 5.0) * epoch_count
    return (
        estimated_training_only_seconds
        + max(0.0, model_load_seconds)
        + eval_overhead
        + checkpoint_overhead
    )


def estimate_gpu_hours(*, elapsed_seconds: float | None, gpu_count: int) -> float | None:
    if elapsed_seconds is None or elapsed_seconds < 0:
        return None
    if gpu_count < 1:
        raise ValueError("gpu_count must be >= 1")
    return (elapsed_seconds / 3600.0) * gpu_count


def resolve_gpu_count(cli_gpu_count: int | None = None) -> int:
    if cli_gpu_count is not None:
        if cli_gpu_count < 1:
            raise ValueError("GPU count must be >= 1")
        return cli_gpu_count
    for key in ("TRAIN_GPU_COUNT", "SLURM_GPUS_ON_NODE", "SLURM_GPUS"):
        raw = os.environ.get(key)
        if raw and str(raw).strip():
            # SLURM_GPUS may be like "1" or "gpu:1"
            digits = "".join(ch for ch in str(raw) if ch.isdigit())
            if digits:
                value = int(digits)
                if value < 1:
                    raise ValueError("GPU count must be >= 1")
                return value
    return 1


def get_git_commit_sha(repo_root: Path | None = None) -> str | None:
    root = repo_root or Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    sha = completed.stdout.strip()
    return sha or None


def get_slurm_job_id() -> str | None:
    for key in ("SLURM_JOB_ID", "SLURM_JOBID"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    return None


def build_runtime_report(
    *,
    mode: str,
    model_id: str,
    gpu_count: int,
    train_example_count: int,
    validation_example_count: int,
    epochs: float,
    effective_batch: int,
    estimated_optimizer_steps: int,
    completed_steps: int,
    intended_optimizer_steps: int | None = None,
    missing_optimizer_steps: int | None = None,
    completed_step_ratio: float | None = None,
    training_length_satisfied: bool | None = None,
    model_load_seconds: float,
    training_seconds: float,
    evaluation_seconds: float,
    total_elapsed_seconds: float,
    average_seconds_per_step_value: float | None,
    average_seconds_per_step_excluding_first: float | None,
    estimated_training_only_seconds: float | None,
    estimated_conservative_total_seconds: float | None,
    estimated_gpu_hours: float | None,
    actual_gpu_hours: float | None,
    git_commit_sha: str | None,
    slurm_job_id: str | None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "modelId": model_id,
        "gpuCount": gpu_count,
        "trainExampleCount": train_example_count,
        "validationExampleCount": validation_example_count,
        "epochs": epochs,
        "effectiveBatchSize": effective_batch,
        "estimatedOptimizerSteps": estimated_optimizer_steps,
        "completedSteps": completed_steps,
        # Optimizer steps this run was configured to execute (max_steps).
        "intendedOptimizerSteps": (
            estimated_optimizer_steps
            if intended_optimizer_steps is None
            else intended_optimizer_steps
        ),
        "missingOptimizerSteps": missing_optimizer_steps,
        "completedStepRatio": completed_step_ratio,
        "trainingLengthSatisfied": training_length_satisfied,
        "modelLoadSeconds": model_load_seconds,
        "trainingSeconds": training_seconds,
        "evaluationSeconds": evaluation_seconds,
        "totalElapsedSeconds": total_elapsed_seconds,
        "averageSecondsPerStep": average_seconds_per_step_value,
        "averageSecondsPerStepExcludingFirst": average_seconds_per_step_excluding_first,
        "estimatedTrainingOnlySeconds": estimated_training_only_seconds,
        "estimatedConservativeTotalSeconds": estimated_conservative_total_seconds,
        "estimatedGpuHours": estimated_gpu_hours,
        "actualGpuHours": actual_gpu_hours,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gitCommitSha": git_commit_sha,
        "slurmJobId": slurm_job_id,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.2f}m ({seconds:.1f}s)"
    hours = seconds / 3600.0
    return f"{hours:.2f}h ({seconds:.1f}s)"


def print_smoke_benchmark(report: dict[str, Any]) -> None:
    print("\nSmoke benchmark:")
    print(f"  - Completed steps: {report.get('completedSteps')}")
    print(f"  - Training time: {format_duration(report.get('trainingSeconds'))}")
    print(
        "  - Average seconds per step: "
        f"{report.get('averageSecondsPerStep') if report.get('averageSecondsPerStep') is not None else 'n/a'}"
    )
    print(f"  - Estimated full optimizer steps: {report.get('estimatedOptimizerSteps')}")
    print(
        "  - Estimated training-only duration: "
        f"{format_duration(report.get('estimatedTrainingOnlySeconds'))}"
    )
    print(
        "  - Conservative estimated total duration: "
        f"{format_duration(report.get('estimatedConservativeTotalSeconds'))}"
    )
    print(f"  - Requested GPUs: {report.get('gpuCount')}")
    gpu_hours = report.get("estimatedGpuHours")
    if isinstance(gpu_hours, (int, float)):
        print(f"  - Estimated GPU hours: {gpu_hours:.4f}")
    else:
        print("  - Estimated GPU hours: n/a")
    print(
        "  Note: estimates are approximate. One-time model download is excluded from "
        "steady-state training-only estimates; conservative totals add model load, "
        "evaluation, and checkpoint overhead."
    )


def resolve_smoke_limits(
    *,
    smoke_test: bool,
    train_records: list[dict[str, str]],
    validation_records: list[dict[str, str]],
    smoke_train_limit: int = 4,
    smoke_validation_limit: int = 2,
    smoke_max_steps: int = 3,
) -> tuple[list[dict[str, str]], list[dict[str, str]], int | None]:
    """Apply smoke-test subsetting and max_steps. Full runs return unchanged data."""
    if not smoke_test:
        return train_records, validation_records, None
    if not train_records:
        raise TrainingDataError("Training dataset is empty")
    train_subset = train_records[: min(smoke_train_limit, len(train_records))]
    val_subset = validation_records[: min(smoke_validation_limit, len(validation_records))]
    return train_subset, val_subset, smoke_max_steps


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/exports/css-360-winter-2026-a7rp/train.jsonl"),
        help="Path to train.jsonl",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/exports/css-360-winter-2026-a7rp/validation.jsonl"),
        help="Path to validation.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/outputs/css-360-qlora"),
        help="Directory for adapter, metrics, and runtime report",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=None,
        help="Requested GPU count (defaults to Slurm env or 1)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Tiny run: 4 train / 2 val examples, max_steps=3, still saves adapter",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Training path (requires GPU + ML stack)
# ---------------------------------------------------------------------------


class StepTimingCallback:
    """TrainerCallback stand-in registered after transformers import."""

    def __init__(self) -> None:
        self.step_durations: list[float] = []
        self.completed_steps = 0
        self.evaluation_seconds = 0.0
        self._step_start: float | None = None
        self._eval_start: float | None = None

    def attach(self, callback_base: type) -> Any:
        timing = self

        class _Callback(callback_base):  # type: ignore[misc, valid-type]
            def on_step_begin(self, args, state, control, **kwargs):  # noqa: ANN001
                timing._step_start = time.perf_counter()
                return control

            def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
                if timing._step_start is not None:
                    timing.step_durations.append(time.perf_counter() - timing._step_start)
                    timing.completed_steps += 1
                    timing._step_start = None
                return control

            def on_prediction_step(self, args, state, control, **kwargs):  # noqa: ANN001
                if timing._eval_start is None:
                    timing._eval_start = time.perf_counter()
                return control

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: ANN001
                if timing._eval_start is not None:
                    timing.evaluation_seconds += time.perf_counter() - timing._eval_start
                    timing._eval_start = None
                return control

        return _Callback()


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. QLoRA training requires a GPU "
            "(e.g. submit via training/smoke.slurm or training/train.slurm on Tillicum)."
        )


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    process_start = time.perf_counter()
    require_cuda()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    gpu_count = resolve_gpu_count(args.gpu_count)
    set_seed(args.seed)

    full_train = load_instruction_response_jsonl(args.train_file)
    full_validation = load_instruction_response_jsonl(args.validation_file)
    if not full_train:
        raise TrainingDataError("Training dataset is empty")

    train_records, validation_records, smoke_max_steps = resolve_smoke_limits(
        smoke_test=args.smoke_test,
        train_records=full_train,
        validation_records=full_validation,
    )
    if not train_records:
        raise TrainingDataError("Training dataset is empty after smoke filtering")

    mode = "smoke" if args.smoke_test else "full"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full runs pass an explicit ceil-based optimizer-step budget so Trainer's
    # floor(len(dataloader) / gradient_accumulation_steps) does not truncate the
    # run when the dataset is not divisible by the effective batch size.
    resolved_max_steps = resolve_max_steps(
        smoke_test=bool(args.smoke_test),
        smoke_max_steps=smoke_max_steps,
        train_example_count=len(train_records),
        epochs=args.epochs,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gpu_count=gpu_count,
    )

    resolved = ResolvedRunConfig(
        mode=mode,
        model_id=args.model_id,
        train_path=str(args.train_file),
        validation_path=str(args.validation_file),
        output_dir=str(output_dir),
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
        gpu_count=gpu_count,
        smoke_test=bool(args.smoke_test),
        max_steps=resolved_max_steps,
        train_example_count=len(train_records),
        validation_example_count=len(validation_records),
        full_train_example_count=len(full_train),
        full_validation_example_count=len(full_validation),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    write_json(output_dir / "resolved_config.json", asdict(resolved))

    effective_batch = effective_batch_size(
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gpu_count=gpu_count,
    )
    full_optimizer_steps = estimate_optimizer_steps(
        train_example_count=len(full_train),
        epochs=args.epochs,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gpu_count=gpu_count,
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"Loading tokenizer/model: {args.model_id}")
    model_load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGET_MODULES),
    )
    model = get_peft_model(model, lora_config)
    model_load_seconds = time.perf_counter() - model_load_start
    print(f"Model load time (excluding prior HF download waits if cached): {model_load_seconds:.2f}s")

    def to_chat_text(example: dict[str, str]) -> dict[str, str]:
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_dataset = Dataset.from_list(train_records).map(to_chat_text)
    eval_dataset = (
        Dataset.from_list(validation_records).map(to_chat_text)
        if validation_records
        else None
    )

    timing = StepTimingCallback()
    timing_callback = timing.attach(TrainerCallback)

    sft_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": 1,
        "eval_strategy": "epoch" if eval_dataset is not None else "no",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "gradient_checkpointing": True,
        "report_to": [],
        "seed": args.seed,
        "max_seq_length": args.max_seq_length,
        "packing": False,
        "dataset_text_field": "text",
    }
    # Both modes supply an explicit max_steps; only smoke runs switch to
    # step-based eval/save, so full-run eval/save stay per-epoch as before.
    sft_kwargs["max_steps"] = resolved_max_steps
    if smoke_max_steps is not None:
        # Keep periodic eval/save usable during short smoke runs.
        sft_kwargs["eval_strategy"] = "steps" if eval_dataset is not None else "no"
        sft_kwargs["eval_steps"] = max(1, resolved_max_steps)
        sft_kwargs["save_strategy"] = "steps"
        sft_kwargs["save_steps"] = max(1, resolved_max_steps)

    sft_config = SFTConfig(**sft_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "callbacks": [timing_callback],
    }
    # TRL version compatibility: newer uses processing_class, older uses tokenizer.
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    training_start = time.perf_counter()
    train_result = trainer.train()
    training_end = time.perf_counter()
    training_seconds = training_end - training_start

    eval_metrics: dict[str, Any] = {}
    eval_start = time.perf_counter()
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
    evaluation_seconds = timing.evaluation_seconds + (time.perf_counter() - eval_start)

    adapter_dir = output_dir / "adapter"
    tokenizer_dir = output_dir / "tokenizer"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    trainer.save_state()
    # Promote trainer_state.json to the run root when Transformers wrote it under checkpoints/.
    for candidate in (
        output_dir / "checkpoints" / "trainer_state.json",
        output_dir / "trainer_state.json",
    ):
        if candidate.is_file():
            if candidate.parent != output_dir:
                target = output_dir / "trainer_state.json"
                target.write_text(candidate.read_text(encoding="utf-8"), encoding="utf-8")
            break

    train_metrics = dict(train_result.metrics)
    train_metrics["train_runtime_seconds_wall"] = training_seconds
    write_json(output_dir / "training_metrics.json", train_metrics)
    write_json(output_dir / "evaluation_metrics.json", eval_metrics)

    completed_steps = max(
        timing.completed_steps,
        int(train_metrics.get("train_steps", 0) or 0),
        int(getattr(trainer.state, "global_step", 0) or 0),
    )
    length_status = evaluate_training_length(
        completed_steps=completed_steps,
        intended_steps=resolved_max_steps,
    )
    all_avg = average_seconds_per_step(timing.step_durations, exclude_first=False)
    skip_first_avg = average_seconds_per_step(timing.step_durations, exclude_first=True)
    conservative_avg = choose_conservative_step_average(all_avg, skip_first_avg)

    estimated_training_only = None
    estimated_conservative = None
    estimated_gpu = None
    if mode == "smoke":
        estimated_training_only = estimate_training_only_seconds(
            average_seconds_per_step_value=conservative_avg,
            estimated_optimizer_steps=full_optimizer_steps,
        )
        estimated_conservative = estimate_conservative_total_seconds(
            estimated_training_only_seconds=estimated_training_only,
            model_load_seconds=model_load_seconds,
            evaluation_seconds=evaluation_seconds,
            epochs=args.epochs,
            average_seconds_per_step_value=conservative_avg,
        )
        estimated_gpu = estimate_gpu_hours(
            elapsed_seconds=estimated_conservative,
            gpu_count=gpu_count,
        )

    total_elapsed = time.perf_counter() - process_start
    actual_gpu_hours = None
    if mode == "full":
        actual_gpu_hours = estimate_gpu_hours(
            elapsed_seconds=total_elapsed,
            gpu_count=gpu_count,
        )

    report = build_runtime_report(
        mode=mode,
        model_id=args.model_id,
        gpu_count=gpu_count,
        train_example_count=len(train_records),
        validation_example_count=len(validation_records),
        epochs=args.epochs,
        effective_batch=effective_batch,
        estimated_optimizer_steps=full_optimizer_steps,
        completed_steps=completed_steps,
        intended_optimizer_steps=resolved_max_steps,
        missing_optimizer_steps=length_status["missingOptimizerSteps"],
        completed_step_ratio=length_status["completedStepRatio"],
        training_length_satisfied=length_status["trainingLengthSatisfied"],
        model_load_seconds=model_load_seconds,
        training_seconds=training_seconds,
        evaluation_seconds=evaluation_seconds,
        total_elapsed_seconds=total_elapsed,
        average_seconds_per_step_value=conservative_avg,
        average_seconds_per_step_excluding_first=skip_first_avg,
        estimated_training_only_seconds=estimated_training_only,
        estimated_conservative_total_seconds=estimated_conservative,
        estimated_gpu_hours=estimated_gpu,
        actual_gpu_hours=actual_gpu_hours,
        git_commit_sha=get_git_commit_sha(),
        slurm_job_id=get_slurm_job_id(),
    )
    # Also record raw averages for transparency.
    report["averageSecondsPerStepAll"] = all_avg
    report["firstStepSeconds"] = timing.step_durations[0] if timing.step_durations else None
    write_json(output_dir / "runtime-report.json", report)

    if mode == "smoke":
        print_smoke_benchmark(report)
    else:
        print("\nFull run complete:")
        print(
            f"  - Optimizer steps: {completed_steps}/{resolved_max_steps} "
            f"(requested epochs: {args.epochs})"
        )
        print(f"  - Total elapsed: {format_duration(total_elapsed)}")
        if actual_gpu_hours is not None:
            print(f"  - Actual GPU hours: {actual_gpu_hours:.4f}")
        else:
            print("  - Actual GPU hours: n/a")

    print(f"\nWrote adapter to {adapter_dir}")
    print(f"Wrote runtime report to {output_dir / 'runtime-report.json'}")

    # A full run must not silently finish materially short of its step budget.
    if mode == "full" and not length_status["trainingLengthSatisfied"]:
        raise RuntimeError(
            "Full training run finished short of its intended length: "
            f"{length_status['completedSteps']}/{length_status['intendedOptimizerSteps']}"
            " optimizer steps "
            f"({length_status['missingOptimizerSteps']} missing). "
            "See runtime-report.json for details."
        )
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_training(args)
    except (TrainingDataError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
