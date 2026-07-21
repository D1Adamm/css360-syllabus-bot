#!/usr/bin/env python3
"""Compare base Llama 3.2 3B Instruct vs CSS 360 QLoRA adapter on held-out questions.

Loads models sequentially on one GPU (base, then adapter) to limit memory use.
Does not merge the adapter into the base weights.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_SEED = 360
DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.72


class ComparisonError(ValueError):
    """Raised when comparison inputs are invalid."""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


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


def normalize_question(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def tokenize_question(text: str) -> set[str]:
    tokens = normalize_question(text).split()
    return {token for token in tokens if token}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def load_reference_instructions(paths: list[str | Path]) -> list[str]:
    instructions: list[str] = []
    for path in paths:
        file_path = Path(path)
        if not file_path.is_file():
            raise ComparisonError(f"Reference file does not exist: {file_path}")
        for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                raise ComparisonError(f"Blank line in {file_path} at line {line_number}")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ComparisonError(
                    f"Malformed JSONL in {file_path} at line {line_number}: {exc.msg}"
                ) from exc
            instruction = payload.get("instruction") if isinstance(payload, dict) else None
            if not isinstance(instruction, str) or not instruction.strip():
                raise ComparisonError(
                    f"Missing instruction in {file_path} at line {line_number}"
                )
            instructions.append(instruction.strip())
    return instructions


def load_heldout_questions(path: str | Path) -> list[str]:
    file_path = Path(path)
    if not file_path.is_file():
        raise ComparisonError(f"Held-out questions file does not exist: {file_path}")
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        raw_questions = payload["questions"]
    elif isinstance(payload, list):
        raw_questions = payload
    else:
        raise ComparisonError(
            "heldout_questions.json must be a list of strings or "
            '{"questions": [...]}'
        )

    questions: list[str] = []
    for index, item in enumerate(raw_questions, start=1):
        if isinstance(item, str):
            question = item.strip()
        elif isinstance(item, dict) and isinstance(item.get("question"), str):
            question = item["question"].strip()
        else:
            raise ComparisonError(f"Invalid held-out question at index {index}")
        if not question:
            raise ComparisonError(f"Blank held-out question at index {index}")
        questions.append(question)

    if not (8 <= len(questions) <= 10):
        raise ComparisonError(
            f"Expected 8–10 held-out questions, found {len(questions)}"
        )
    return questions


def find_exact_training_overlap(question: str, reference_instructions: list[str]) -> bool:
    normalized = normalize_question(question)
    return any(normalize_question(ref) == normalized for ref in reference_instructions)


def find_near_duplicate_warning(
    question: str,
    reference_instructions: list[str],
    *,
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> str | None:
    question_tokens = tokenize_question(question)
    best_score = 0.0
    best_ref = ""
    for ref in reference_instructions:
        score = jaccard_similarity(question_tokens, tokenize_question(ref))
        if score > best_score:
            best_score = score
            best_ref = ref
    if best_score >= threshold:
        return (
            f"Near-duplicate similarity {best_score:.2f} vs training/validation "
            f"instruction: {best_ref}"
        )
    return None


def validate_heldout_against_references(
    questions: list[str],
    reference_instructions: list[str],
    *,
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Reject exact overlaps; attach near-duplicate warnings for soft matches."""
    prepared: list[dict[str, Any]] = []
    for question in questions:
        exact = find_exact_training_overlap(question, reference_instructions)
        if exact:
            raise ComparisonError(
                "Held-out question exactly matches a train/validation instruction: "
                f"{question}"
            )
        warning = find_near_duplicate_warning(
            question,
            reference_instructions,
            threshold=near_duplicate_threshold,
        )
        prepared.append(
            {
                "question": question,
                "exactTrainingOverlap": False,
                "nearDuplicateWarning": warning,
            }
        )
    return prepared


def build_comparison_summary(
    *,
    model_id: str,
    adapter_path: str,
    question_count: int,
    total_base_generation_seconds: float,
    total_finetuned_generation_seconds: float,
    git_commit_sha: str | None,
    slurm_job_id: str | None,
) -> dict[str, Any]:
    return {
        "modelId": model_id,
        "adapterPath": adapter_path,
        "questionCount": question_count,
        "totalBaseGenerationSeconds": total_base_generation_seconds,
        "totalFineTunedGenerationSeconds": total_finetuned_generation_seconds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gitCommitSha": git_commit_sha,
        "slurmJobId": slurm_job_id,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=Path(
            f"/gpfs/projects/simswe/{os.environ.get('USER', 'USER')}/training_outputs/css-360-qlora/adapter"
        ),
    )
    parser.add_argument(
        "--heldout-file",
        type=Path,
        default=Path("training/heldout_questions.json"),
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/exports/css-360-winter-2026-a7rp/train.jsonl"),
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/exports/css-360-winter-2026-a7rp/validation.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/outputs/css-360-comparison"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    )
    return parser.parse_args(argv)


def _apply_chat_template(tokenizer: Any, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _model_device(model: Any) -> Any:
    try:
        return model.device
    except Exception:  # noqa: BLE001
        return next(model.parameters()).device


def _generate_answer(
    *,
    model: Any,
    tokenizer: Any,
    question: str,
    max_new_tokens: int,
    repetition_penalty: float,
    seed: int,
) -> tuple[str, float]:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prompt = _apply_chat_template(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = _model_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    elapsed = time.perf_counter() - start

    prompt_length = inputs["input_ids"].shape[-1]
    completion_ids = output_ids[0, prompt_length:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return text, elapsed


def _load_base_model(model_id: str, tokenizer: Any) -> Any:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model.eval()
    return model


def _release_model(model: Any) -> None:
    import gc

    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; comparison requires one GPU.")

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise ComparisonError(f"Adapter path does not exist: {adapter_path}")

    reference_instructions = load_reference_instructions(
        [args.train_file, args.validation_file]
    )
    heldout_questions = load_heldout_questions(args.heldout_file)
    prepared = validate_heldout_against_references(
        heldout_questions,
        reference_instructions,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = [
        {
            "question": item["question"],
            "baseResponse": "",
            "fineTunedResponse": "",
            "baseGenerationSeconds": 0.0,
            "fineTunedGenerationSeconds": 0.0,
            "exactTrainingOverlap": item["exactTrainingOverlap"],
            "nearDuplicateWarning": item["nearDuplicateWarning"],
        }
        for item in prepared
    ]

    print("Loading base model...")
    base_model = _load_base_model(args.model_id, tokenizer)
    total_base = 0.0
    for index, item in enumerate(results):
        print(f"[base] {index + 1}/{len(results)}: {item['question'][:80]}")
        answer, elapsed = _generate_answer(
            model=base_model,
            tokenizer=tokenizer,
            question=item["question"],
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )
        item["baseResponse"] = answer
        item["baseGenerationSeconds"] = elapsed
        total_base += elapsed
    _release_model(base_model)

    print("Loading fine-tuned adapter (same base + PEFT, not merged)...")
    base_for_adapter = _load_base_model(args.model_id, tokenizer)
    ft_model = PeftModel.from_pretrained(base_for_adapter, str(adapter_path))
    ft_model.eval()
    total_ft = 0.0
    for index, item in enumerate(results):
        print(f"[finetuned] {index + 1}/{len(results)}: {item['question'][:80]}")
        answer, elapsed = _generate_answer(
            model=ft_model,
            tokenizer=tokenizer,
            question=item["question"],
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )
        item["fineTunedResponse"] = answer
        item["fineTunedGenerationSeconds"] = elapsed
        total_ft += elapsed
    _release_model(ft_model)

    summary = build_comparison_summary(
        model_id=args.model_id,
        adapter_path=str(adapter_path),
        question_count=len(results),
        total_base_generation_seconds=total_base,
        total_finetuned_generation_seconds=total_ft,
        git_commit_sha=get_git_commit_sha(),
        slurm_job_id=get_slurm_job_id(),
    )

    write_json(output_dir / "comparison_results.json", results)
    write_jsonl(output_dir / "comparison_results.jsonl", results)
    write_json(output_dir / "comparison_summary.json", summary)

    print(f"Wrote results to {output_dir}")
    return {"results": results, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_comparison(args)
    except (ComparisonError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
