#!/usr/bin/env python3
"""QLoRA fine-tuning for CSS 360 syllabus instruction/response JSONL.

Designed for a single Tillicum GPU via Slurm. Supports a tiny --smoke-test mode
that still measures step timing and estimates full-run duration / GPU hours.

``--cpu`` runs the same recipe on a machine with no GPU (the UWB VM): the same
JSONL, chat formatting, LoRA configuration and 4-bit NF4 quantization, with
float32 compute, no fp16/bf16 autocast, batch size 1 and no CUDA requirement.
The two device modes are explicit and never fall back to each other.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
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
#: 512 was enough for a bare question and a short answer. A grounded example's
#: user turn is the production grounded prompt, which the retrieval budget
#: bounds at roughly 1,600 tokens; 2048 holds that and a 256-token answer.
DEFAULT_MAX_SEQ_LENGTH = 2048
#: Where the answer starts in the Llama 3 chat format. The completion-only
#: collator on older TRL releases masks everything up to and including this.
LLAMA3_RESPONSE_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>\n\n"
#: Every record has a format and a kind; older bare exports carry neither and
#: read as these.
DEFAULT_RECORD_FORMAT = "bare"
DEFAULT_RECORD_KIND = "answerable"
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
DEVICE_MODES = ("cuda", "cpu")
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True
LIBRARY_VERSION_NAMES = (
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "trl",
    "datasets",
    "accelerate",
)
MIB = 1024 * 1024


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
    # Device facts. Defaults describe the original single-GPU recipe so older
    # readers of resolved_config.json see the same shape they always did.
    device: str = "cuda"
    compute_dtype: str = "bfloat16"
    cpu_threads: int | None = None
    gradient_checkpointing: bool = True
    bnb_4bit_quant_type: str = BNB_4BIT_QUANT_TYPE
    bnb_4bit_use_double_quant: bool = BNB_4BIT_USE_DOUBLE_QUANT
    lora_target_modules: tuple[str, ...] = LORA_TARGET_MODULES


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
        record: dict[str, str] = {
            "instruction": instruction,
            "response": response,
        }
        # Mixed-format exports say what each record is; a bare export does not,
        # and reads as bare/answerable. Both train the same way: `instruction`
        # is the user turn, `response` the assistant turn.
        for key in ("format", "kind"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                record[key] = value.strip()
        records.append(record)
    return records


def dataset_composition(records: list[dict[str, str]]) -> dict[str, int]:
    """How many records of each `format/kind`, for the runtime report."""
    counts: dict[str, int] = {}
    for record in records:
        key = (
            f"{record.get('format') or DEFAULT_RECORD_FORMAT}/"
            f"{record.get('kind') or DEFAULT_RECORD_KIND}"
        )
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


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
    device: str = "cuda",
    compute_dtype: str | None = None,
    cpu_threads: int | None = None,
    peak_rss_bytes_value: int | None = None,
    peak_rss_after_model_load_bytes: int | None = None,
    library_versions: dict[str, str | None] | None = None,
    dataset_composition_value: dict[str, Any] | None = None,
    completion_only_strategy: str | None = None,
    max_seq_length: int | None = None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        # What the adapter was trained on: how many records of each
        # format/kind, whether the loss covered the answer only, and the
        # sequence window the grounded prompts had to fit.
        "datasetComposition": dataset_composition_value,
        "completionOnlyLoss": completion_only_strategy,
        "maxSeqLength": max_seq_length,
        "modelId": model_id,
        "device": device,
        "computeDtype": compute_dtype,
        "cpuThreads": cpu_threads,
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
        "peakRssBytes": peak_rss_bytes_value,
        "peakRssMiB": (
            peak_rss_bytes_value / MIB if peak_rss_bytes_value is not None else None
        ),
        "peakRssAfterModelLoadBytes": peak_rss_after_model_load_bytes,
        "libraryVersions": library_versions,
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


def print_peak_memory(report: dict[str, Any]) -> None:
    peak = report.get("peakRssMiB")
    if isinstance(peak, (int, float)):
        print(f"  - Peak resident memory: {peak:.0f} MiB")


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
    if report.get("device") == "cpu":
        print(
            f"  - Device: cpu ({report.get('cpuThreads')} torch threads, float32 compute)"
        )
    else:
        print(f"  - Requested GPUs: {report.get('gpuCount')}")
        gpu_hours = report.get("estimatedGpuHours")
        if isinstance(gpu_hours, (int, float)):
            print(f"  - Estimated GPU hours: {gpu_hours:.4f}")
        else:
            print("  - Estimated GPU hours: n/a")
    print_peak_memory(report)
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
    train_subset = smoke_subset(train_records, smoke_train_limit)
    val_subset = smoke_subset(validation_records, smoke_validation_limit)
    return train_subset, val_subset, smoke_max_steps


def smoke_subset(records: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    """The first `limit` records, but with every format represented.

    A smoke run exists to exercise the real training path before an hour is
    spent on it. On a mixed dataset that means at least one grounded record
    (the long prompt, the masked loss) and at least one bare record must be in
    the subset, whatever the shuffle put first. The first record of each
    format is taken, then the earliest remaining records fill the limit, in
    their original order.
    """
    if limit <= 0 or not records:
        return []
    chosen: list[int] = []
    seen_formats: set[str] = set()
    for index, record in enumerate(records):
        fmt = record.get("format") or DEFAULT_RECORD_FORMAT
        if fmt not in seen_formats:
            seen_formats.add(fmt)
            chosen.append(index)
    chosen = chosen[:limit]
    for index in range(len(records)):
        if len(chosen) >= limit:
            break
        if index not in chosen:
            chosen.append(index)
    return [records[index] for index in sorted(chosen)]


# ---------------------------------------------------------------------------
# Device mode (pure; no torch import)
#
# "cuda" is the original Tillicum recipe and is untouched. "cpu" is chosen only
# by an explicit --cpu flag: nothing here inspects the machine to pick a mode,
# and neither mode falls back to the other. A CUDA run without a GPU still
# fails in require_cuda(); a CPU run never touches CUDA even if one exists.
# ---------------------------------------------------------------------------


def resolve_device(cpu: bool) -> str:
    return "cpu" if cpu else "cuda"


def validate_device_arguments(args: argparse.Namespace) -> None:
    """Reject option mixes that would otherwise have to be guessed at."""
    cpu = bool(getattr(args, "cpu", False))
    cpu_threads = getattr(args, "cpu_threads", None)
    if not cpu:
        if cpu_threads is not None:
            raise ValueError("--cpu-threads is only valid together with --cpu")
        return
    if getattr(args, "gpu_count", None) is not None:
        raise ValueError(
            "--cpu and --gpu-count are mutually exclusive: CPU mode uses no GPU"
        )
    if int(args.per_device_batch_size) != 1:
        raise ValueError("CPU mode requires --per-device-batch-size 1")
    if cpu_threads is not None and int(cpu_threads) < 1:
        raise ValueError("--cpu-threads must be >= 1")


def quantization_settings(compute_dtype: Any) -> dict[str, Any]:
    """BitsAndBytesConfig kwargs. Identical in both modes except the compute dtype."""
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": BNB_4BIT_QUANT_TYPE,
        "bnb_4bit_use_double_quant": BNB_4BIT_USE_DOUBLE_QUANT,
        "bnb_4bit_compute_dtype": compute_dtype,
    }


def model_load_settings(device: str, compute_dtype: Any) -> dict[str, Any]:
    """from_pretrained placement kwargs.

    CPU mode needs a *dict* device map whose every value is "cpu": that is the
    exact shape the transformers 4-bit quantizer accepts when the bitsandbytes
    multi-backend is present. "auto" would dispatch to whatever accelerate
    finds, which is the silent fallback this trainer refuses to have.
    """
    if device == "cpu":
        return {"device_map": {"": "cpu"}, "torch_dtype": compute_dtype}
    if device == "cuda":
        return {"device_map": "auto", "torch_dtype": compute_dtype}
    raise ValueError(f"Unknown device mode: {device!r}")


def precision_settings(device: str, use_bf16: bool) -> dict[str, Any]:
    """Trainer precision flags. CPU: float32 throughout, no autocast, no pinning."""
    if device == "cpu":
        if use_bf16:
            raise ValueError("bf16 autocast is not used in CPU mode")
        return {
            "bf16": False,
            "fp16": False,
            "use_cpu": True,
            "dataloader_pin_memory": False,
        }
    if device == "cuda":
        return {"bf16": use_bf16, "fp16": not use_bf16}
    raise ValueError(f"Unknown device mode: {device!r}")


def dtype_name(dtype: Any) -> str:
    """``torch.float32`` -> ``"float32"`` for JSON metadata."""
    text = str(dtype)
    return text[len("torch."):] if text.startswith("torch.") else text


def format_chat_example(tokenizer: Any, example: dict[str, str]) -> dict[str, str]:
    """One instruction/response pair as the model's own chat format.

    Shared by both device modes so the CPU path trains on byte-identical text.
    """
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


def format_prompt_completion(tokenizer: Any, example: dict[str, str]) -> dict[str, str]:
    """The same chat text as `format_chat_example`, split where the answer starts.

    `prompt` is the user turn rendered by the chat template with the
    generation header appended; `completion` is the answer closed with the
    end-of-turn token. Concatenated they are byte-identical to the single
    `text` the older format produced, which `assert_prompt_completion_matches`
    checks against the real tokenizer before training. TRL trains on the
    completion only when given the two halves.
    """
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["instruction"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    end_of_turn = getattr(tokenizer, "eos_token", None) or ""
    return {"prompt": prompt, "completion": example["response"] + end_of_turn}


def assert_prompt_completion_matches(tokenizer: Any) -> None:
    """Refuse to train if the two renderings disagree for this tokenizer."""
    probe = {"instruction": "probe question?", "response": "probe answer."}
    halves = format_prompt_completion(tokenizer, probe)
    whole = format_chat_example(tokenizer, probe)["text"]
    if halves["prompt"] + halves["completion"] != whole:
        raise TrainingDataError(
            "The chat template's assistant turn does not split into prompt and "
            "completion as expected, so completion-only loss would train on a "
            "different text than the whole-text format. Check the tokenizer's "
            "chat template and eos_token."
        )


def resolve_completion_only_strategy(config_class: Any) -> str:
    """How this TRL release masks the prompt out of the loss.

    `config` when SFTConfig accepts `completion_only_loss` (TRL 0.19 and later,
    the VM's 1.12): the dataset carries `prompt` and `completion` columns and
    TRL masks the prompt itself. `collator` otherwise (TRL 0.13, the cluster
    pins): the dataset carries the whole `text` and
    `DataCollatorForCompletionOnlyLM` masks everything up to the answer's
    header. Either way the loss covers the answer alone; a grounded prompt of
    a thousand tokens of syllabus is never a training target.
    """
    supported = supported_config_parameters(config_class)
    if supported is None or "completion_only_loss" in supported:
        return "config"
    return "collator"


def completion_only_config(strategy: str) -> dict[str, Any]:
    if strategy == "config":
        return {"completion_only_loss": True}
    if strategy == "collator":
        return {"dataset_text_field": "text"}
    raise ValueError(f"Unknown completion-only strategy: {strategy!r}")


def training_row(tokenizer: Any, example: dict[str, str], *, strategy: str) -> dict[str, str]:
    """One record as the dataset row the chosen strategy trains on."""
    if strategy == "config":
        return format_prompt_completion(tokenizer, example)
    if strategy == "collator":
        return format_chat_example(tokenizer, example)
    raise ValueError(f"Unknown completion-only strategy: {strategy!r}")


def build_completion_only_collator(tokenizer: Any) -> Any:
    """The older TRL's prompt mask, keyed on the answer header's token ids."""
    from trl import DataCollatorForCompletionOnlyLM

    template_ids = tokenizer.encode(LLAMA3_RESPONSE_TEMPLATE, add_special_tokens=False)
    return DataCollatorForCompletionOnlyLM(response_template=template_ids, tokenizer=tokenizer)


def supported_config_parameters(config_class: Any) -> set[str] | None:
    """Names ``config_class(...)`` accepts, or None when it takes ``**kwargs``.

    A dataclass such as SFTConfig lists every field, inherited ones included,
    in its ``__init__`` signature, so for the real class this is exact. A
    ``**kwargs`` signature only occurs on test stand-ins, which accept anything
    by construction.
    """
    parameters = inspect.signature(config_class).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return None
    return {name for name in parameters if name != "self"}


def warmup_steps_from_ratio(warmup_ratio: float, total_optimizer_steps: int) -> int:
    """``warmup_ratio`` as an exact step count, as transformers 4.x resolved it.

    ``TrainingArguments.get_warmup_steps`` computed
    ``ceil(num_training_steps * warmup_ratio)``; this is the same expression,
    so a run on a release without ``warmup_ratio`` gets the schedule the
    cluster used: 1 warmup step of 3 for a smoke run, 2 of 18 for the CSS 360
    split.
    """
    if not 0.0 <= float(warmup_ratio) <= 1.0:
        raise ValueError(f"warmup_ratio must be within [0, 1]: {warmup_ratio!r}")
    if int(total_optimizer_steps) < 1:
        raise ValueError("total_optimizer_steps must be >= 1")
    return int(math.ceil(int(total_optimizer_steps) * float(warmup_ratio)))


def build_sft_config(
    sft_config_class: Any,
    kwargs: dict[str, Any],
    *,
    total_optimizer_steps: int,
) -> Any:
    """Construct SFTConfig from the pinned-era argument names on any release.

    The Tillicum pins (transformers 4.47.1, TRL 0.13) accept every name in
    ``kwargs`` and receive them untouched. Newer releases renamed two: TRL
    0.20 replaced ``max_seq_length`` with ``max_length``, and transformers 5
    replaced ``warmup_ratio`` with ``warmup_steps``. Each is translated only
    when the constructor's own signature lacks the old name and has the new
    one, and the warmup ratio is converted to the step count the old
    scheduler would have derived from it. Any other argument the signature
    does not accept is an error that names it, never a silent drop: a
    hyperparameter that vanished would be a different run.
    """
    supported = supported_config_parameters(sft_config_class)
    if supported is None:
        return sft_config_class(**kwargs)

    adapted = dict(kwargs)
    notes: list[str] = []
    if (
        "max_seq_length" in adapted
        and "max_seq_length" not in supported
        and "max_length" in supported
    ):
        adapted["max_length"] = adapted.pop("max_seq_length")
        notes.append(f"max_seq_length -> max_length={adapted['max_length']}")
    if (
        "warmup_ratio" in adapted
        and "warmup_ratio" not in supported
        and "warmup_steps" in supported
    ):
        ratio = adapted.pop("warmup_ratio")
        adapted["warmup_steps"] = warmup_steps_from_ratio(ratio, total_optimizer_steps)
        notes.append(
            f"warmup_ratio={ratio} -> warmup_steps={adapted['warmup_steps']} "
            f"of {total_optimizer_steps} optimizer steps"
        )

    unsupported = sorted(name for name in adapted if name not in supported)
    if unsupported:
        raise ValueError(
            f"{sft_config_class.__name__} from {sft_config_class.__module__} does not "
            f"accept: {', '.join(unsupported)}. Extend build_sft_config() with the "
            "equivalent for this release rather than dropping them."
        )
    if notes:
        print("SFTConfig compatibility: " + "; ".join(notes))
    return sft_config_class(**adapted)


def peak_rss_bytes() -> int | None:
    """Peak resident set size of this process, in bytes (None if unavailable)."""
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes; macOS reports bytes.
    if sys.platform == "darwin":
        return int(usage)
    return int(usage) * 1024


def collect_library_versions(
    names: tuple[str, ...] = LIBRARY_VERSION_NAMES,
) -> dict[str, str | None]:
    """Versions of the ML stack a run actually used (the CPU venv is not pinned)."""
    versions: dict[str, str | None] = {}
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            try:
                module = importlib.import_module(name)
            except Exception:  # noqa: BLE001 - absent or broken package
                module = None
        version = getattr(module, "__version__", None) if module is not None else None
        versions[name] = str(version) if version else None
    return versions


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
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help=(
            "Tokens per example, prompt and answer together. Grounded examples "
            "render the production prompt (about 1,000-1,300 tokens) and were "
            "unrepresentable at the old 512."
        ),
    )
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
    parser.add_argument(
        "--cpu",
        action="store_true",
        help=(
            "Train on this machine's CPU: no CUDA, float32 compute, no fp16/bf16, "
            "batch size 1. Explicit only; never chosen automatically."
        ),
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="torch intra-op threads in CPU mode (default: torch's own default)",
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
    device = resolve_device(bool(getattr(args, "cpu", False)))
    validate_device_arguments(args)
    if device == "cuda":
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

    if device == "cuda":
        gpu_count = resolve_gpu_count(args.gpu_count)
        device_count = gpu_count
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        cpu_threads = None
    else:
        # Explicit CPU mode: no CUDA, float32 compute, no autocast. A CUDA
        # device that happens to exist is left alone rather than used.
        if torch.cuda.is_available():
            print("NOTE: --cpu given; the available CUDA device will not be used.")
        gpu_count = 0
        device_count = 1
        use_bf16 = False
        compute_dtype = torch.float32
        if args.cpu_threads is not None:
            torch.set_num_threads(int(args.cpu_threads))
        cpu_threads = int(torch.get_num_threads())
        print(f"CPU mode: {cpu_threads} torch threads, float32 compute")
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
        gpu_count=device_count,
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
        device=device,
        compute_dtype=dtype_name(compute_dtype),
        cpu_threads=cpu_threads,
        gradient_checkpointing=True,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
        lora_target_modules=LORA_TARGET_MODULES,
    )
    write_json(output_dir / "resolved_config.json", asdict(resolved))

    effective_batch = effective_batch_size(
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gpu_count=device_count,
    )
    full_optimizer_steps = estimate_optimizer_steps(
        train_example_count=len(full_train),
        epochs=args.epochs,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gpu_count=device_count,
    )

    print(f"Loading tokenizer/model: {args.model_id} (device mode: {device})")
    model_load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(**quantization_settings(compute_dtype))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        **model_load_settings(device, compute_dtype),
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
    peak_rss_after_model_load = peak_rss_bytes()
    if peak_rss_after_model_load is not None:
        print(f"Peak resident memory after model load: {peak_rss_after_model_load / MIB:.0f} MiB")

    completion_strategy = resolve_completion_only_strategy(SFTConfig)
    if completion_strategy == "config":
        assert_prompt_completion_matches(tokenizer)
    print(
        f"Completion-only loss: {completion_strategy} "
        f"({'SFTConfig.completion_only_loss' if completion_strategy == 'config' else 'DataCollatorForCompletionOnlyLM'})"
    )
    train_composition = dataset_composition(train_records)
    validation_composition = dataset_composition(validation_records)
    print(f"Dataset composition: train={train_composition} validation={validation_composition}")

    def to_training_row(example: dict[str, str]) -> dict[str, str]:
        return training_row(tokenizer, example, strategy=completion_strategy)

    train_dataset = Dataset.from_list(train_records).map(to_training_row)
    eval_dataset = (
        Dataset.from_list(validation_records).map(to_training_row)
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
        **precision_settings(device, use_bf16),
        "gradient_checkpointing": True,
        "report_to": [],
        "seed": args.seed,
        "max_seq_length": args.max_seq_length,
        "packing": False,
        **completion_only_config(completion_strategy),
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

    sft_config = build_sft_config(
        SFTConfig, sft_kwargs, total_optimizer_steps=resolved_max_steps
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "callbacks": [timing_callback],
    }
    if completion_strategy == "collator":
        trainer_kwargs["data_collator"] = build_completion_only_collator(tokenizer)
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
        if device == "cuda":
            estimated_gpu = estimate_gpu_hours(
                elapsed_seconds=estimated_conservative,
                gpu_count=gpu_count,
            )

    total_elapsed = time.perf_counter() - process_start
    actual_gpu_hours = None
    if mode == "full" and device == "cuda":
        actual_gpu_hours = estimate_gpu_hours(
            elapsed_seconds=total_elapsed,
            gpu_count=gpu_count,
        )

    report = build_runtime_report(
        dataset_composition_value={
            "train": train_composition,
            "validation": validation_composition,
        },
        completion_only_strategy=completion_strategy,
        max_seq_length=args.max_seq_length,
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
        device=device,
        compute_dtype=dtype_name(compute_dtype),
        cpu_threads=cpu_threads,
        peak_rss_bytes_value=peak_rss_bytes(),
        peak_rss_after_model_load_bytes=peak_rss_after_model_load,
        library_versions=collect_library_versions(),
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
        if device == "cpu":
            print(f"  - Device: cpu ({cpu_threads} torch threads, float32 compute)")
        elif actual_gpu_hours is not None:
            print(f"  - Actual GPU hours: {actual_gpu_hours:.4f}")
        else:
            print("  - Actual GPU hours: n/a")
        print_peak_memory(report)

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
