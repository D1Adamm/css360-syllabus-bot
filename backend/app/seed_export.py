"""Phase 8 local export of reviewed/approved fine-tuning examples.

Writes under data/exports/{courseId}/ and never invents a second Firebase path.
Firebase remains courses/{courseId}/seedExamples only.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.seed_review import is_approved_for_export, resolve_review_status


def project_root() -> Path:
    # backend/app/seed_export.py -> repo root
    return Path(__file__).resolve().parents[2]


def course_export_dir(course_id: str, *, root: Path | None = None) -> Path:
    base = root or project_root()
    return base / "data" / "exports" / course_id


def finetune_record(seed: dict[str, Any]) -> dict[str, str]:
    instruction = str(seed.get("instruction") or seed.get("question") or "").strip()
    response = str(seed.get("response") or seed.get("answer") or "").strip()
    return {"instruction": instruction, "response": response}


def metadata_record(seed: dict[str, Any]) -> dict[str, Any]:
    """Richer artifact for audit / later fine-tuning provenance."""
    return {
        "id": seed.get("id"),
        "instruction": str(seed.get("instruction") or seed.get("question") or "").strip(),
        "response": str(seed.get("response") or seed.get("answer") or "").strip(),
        "question": str(seed.get("question") or seed.get("instruction") or "").strip(),
        "answer": str(seed.get("answer") or seed.get("response") or "").strip(),
        "factId": seed.get("factId"),
        "sourceChunkIds": list(seed.get("sourceChunkIds") or []),
        "evidenceQuote": seed.get("evidenceQuote"),
        "origin": seed.get("origin"),
        "validation": seed.get("validation"),
        "reviewStatus": resolve_review_status(seed),
        "reviewNotes": seed.get("reviewNotes"),
        "originalQuestion": seed.get("originalQuestion"),
        "originalAnswer": seed.get("originalAnswer"),
        "category": seed.get("category"),
        "questionType": seed.get("questionType"),
        "sourceSection": seed.get("sourceSection"),
        "createdAt": seed.get("createdAt"),
        "reviewedAt": seed.get("reviewedAt"),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_approved_seeds(
    *,
    course_id: str,
    seeds: list[dict[str, Any]],
    export_root: Path | None = None,
) -> dict[str, Any]:
    """Export approved-only JSONL + metadata. Skips non-approved seeds."""
    approved = [seed for seed in seeds if is_approved_for_export(seed)]
    out_dir = course_export_dir(course_id, root=export_root)
    finetune_path = out_dir / "approved-finetune.jsonl"
    metadata_path = out_dir / "approved-metadata.json"
    summary_path = out_dir / "approved-export-summary.json"

    finetune_rows = [finetune_record(seed) for seed in approved]
    metadata_rows = [metadata_record(seed) for seed in approved]
    write_jsonl(finetune_path, finetune_rows)
    write_json(metadata_path, metadata_rows)

    skipped = len(seeds) - len(approved)
    by_status: dict[str, int] = {}
    for seed in seeds:
        status = resolve_review_status(seed)
        by_status[status] = by_status.get(status, 0) + 1

    summary = {
        "courseId": course_id,
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "inputCount": len(seeds),
        "approvedCount": len(approved),
        "skippedCount": skipped,
        "reviewStatusCounts": by_status,
        "firebasePath": f"courses/{course_id}/seedExamples",
        "files": {
            "finetuneJsonl": str(finetune_path),
            "metadataJson": str(metadata_path),
            "summaryJson": str(summary_path),
        },
        "note": (
            "Only reviewStatus=approved seeds are exported. "
            "Validated AI seeds remain generated until human review."
        ),
    }
    write_json(summary_path, summary)
    return summary


def write_generation_snapshot(
    *,
    course_id: str,
    seeds: list[dict[str, Any]],
    progress: dict[str, Any] | None = None,
    export_root: Path | None = None,
) -> Path:
    """Local snapshot of a generation run (accepted seeds, still review=generated)."""
    out_dir = course_export_dir(course_id, root=export_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"generated-snapshot-{stamp}.json"
    payload = {
        "courseId": course_id,
        "savedAt": datetime.now(timezone.utc).isoformat(),
        "firebasePath": f"courses/{course_id}/seedExamples",
        "progress": progress or {},
        "seedCount": len(seeds),
        "seeds": [metadata_record(seed) for seed in seeds],
    }
    write_json(path, payload)
    latest = out_dir / "generated-snapshot-latest.json"
    write_json(latest, payload)
    return path
