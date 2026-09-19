"""The lineage record, read for the benchmark route and projected to safe fields.

`evaluation/model_lineage.json` is the machine-readable evidence record of
every CSS 360 artifact: where each adapter was trained, what its bytes hash
to, which Ollama tag holds it. The benchmark route labels every answer with
that record so a saved result says which artifact produced it and not merely
which alias was asked for.

Only a fixed set of fields leaves this module. The record also holds run
directories, log paths, Modelfile lines, cluster paths and hostnames; none of
that is projected, and a new key added to the record cannot widen a response
because every field is copied by name.

No database is involved. The experimental aliases have no registry row, and
the route must not depend on one, so the alias -> lineage id table is code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from app.config import BACKEND_ROOT

CSS360_COURSE_ID = "css-360-winter-2026-a7rp"

#: The fixed experiment aliases, and the control.
CONTROL_ALIAS = "base"
EXPERIMENT_ALIASES: tuple[str, ...] = ("v2", "v3", "v4_vm", "v4_tillicum")
ALIASES: tuple[str, ...] = (CONTROL_ALIAS,) + EXPERIMENT_ALIASES

#: Alias -> `lineageId` in the record. Neither v4 experiment has a version;
#: they are named by lineage id and by alias, never by a version label.
ALIAS_LINEAGE_IDS: Mapping[str, str] = {
    "v2": "css360-v2",
    "v3": "css360-v3",
    "v4_vm": "css360-v4-vm",
    "v4_tillicum": "css360-v4-tillicum",
}

SUPPORTED_SCHEMA_VERSION = 1

DEFAULT_LINEAGE_PATH = BACKEND_ROOT.parent / "evaluation" / "model_lineage.json"


class LineageUnavailable(Exception):
    """The record cannot be read, is not the CSS 360 record, or lacks an artifact.

    The message is safe to show a caller: it names what is missing, never a
    path.
    """


_cache: dict[str, dict[str, Any]] = {}


def reset_lineage_cache() -> None:
    _cache.clear()


def _validate(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise LineageUnavailable("The lineage record is not a JSON object.")
    if record.get("schemaVersion") != SUPPORTED_SCHEMA_VERSION:
        raise LineageUnavailable(
            "The lineage record has an unsupported schema version "
            f"(expected {SUPPORTED_SCHEMA_VERSION})."
        )
    if record.get("courseId") != CSS360_COURSE_ID:
        raise LineageUnavailable("The lineage record is not the CSS 360 record.")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not all(isinstance(a, dict) for a in artifacts):
        raise LineageUnavailable("The lineage record has no artifact list.")
    if not isinstance(record.get("baseModel"), dict):
        raise LineageUnavailable("The lineage record has no base model entry.")
    return record


def load_lineage(path: Path | str | None = None) -> dict[str, Any]:
    """The validated record, cached per path for the life of the process."""
    resolved = Path(path) if path is not None else DEFAULT_LINEAGE_PATH
    key = str(resolved)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    try:
        raw = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise LineageUnavailable("The lineage record could not be read.") from exc
    try:
        record = json.loads(raw)
    except ValueError as exc:
        raise LineageUnavailable("The lineage record is not valid JSON.") from exc
    validated = _validate(record)
    _cache[key] = validated
    return validated


def lineage_record_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """Which record labelled the answers: title, schema, generation time."""
    return {
        "title": _string(record.get("title")),
        "schemaVersion": record.get("schemaVersion"),
        "generatedAt": _string(record.get("generatedAt")),
        "courseId": _string(record.get("courseId")),
    }


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _hex_digest(value: Any) -> str | None:
    """A bare hex SHA-256, with Ollama's `sha256-` blob prefix removed."""
    text = _string(value)
    if text is None:
        return None
    for prefix in ("sha256-", "sha256:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text or None


def _artifact(record: Mapping[str, Any], lineage_id: str) -> dict[str, Any]:
    for artifact in record.get("artifacts") or ():
        if isinstance(artifact, dict) and artifact.get("lineageId") == lineage_id:
            return artifact
    raise LineageUnavailable(f'The lineage record has no artifact "{lineage_id}".')


#: Every key a projected lineage summary may carry. A response is built from
#: this list and from nothing else in the record.
LINEAGE_SUMMARY_FIELDS: tuple[str, ...] = (
    "alias",
    "lineageId",
    "version",
    "role",
    "trainingLabel",
    "includeInControlledClaims",
    "expectedTag",
    "ollamaDigest",
    "baseQuantization",
    "huggingFaceId",
    "adapterSha256",
    "adapterDtype",
    "ggufBlobSha256",
    "ggufDtype",
    "ggufBytes",
    "gitCommitSha",
)


def lineage_for_alias(alias: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """The safe projection of one alias's artifact.

    `ggufBlobSha256` is Ollama's adapter blob digest: the SHA-256 of the GGUF
    file as Ollama imported it, which is the only GGUF hash the record holds.
    `adapterSha256` is the SHA-256 of the saved PEFT `adapter_model.safetensors`.
    """
    if alias not in ALIASES:
        raise LineageUnavailable(f'Unknown benchmark alias "{alias}".')

    if alias == CONTROL_ALIAS:
        base = record.get("baseModel") or {}
        summary: dict[str, Any] = {
            "alias": alias,
            "lineageId": None,
            "version": None,
            "role": "base",
            "trainingLabel": None,
            "includeInControlledClaims": None,
            "expectedTag": _string(base.get("ollamaTag")),
            "ollamaDigest": _string(base.get("ollamaDigest")),
            "baseQuantization": _string(base.get("quantization")),
            "huggingFaceId": _string(base.get("huggingFaceId")),
            "adapterSha256": None,
            "adapterDtype": None,
            "ggufBlobSha256": None,
            "ggufDtype": None,
            "ggufBytes": None,
            "gitCommitSha": None,
        }
    else:
        artifact = _artifact(record, ALIAS_LINEAGE_IDS[alias])
        adapter = artifact.get("adapter") if isinstance(artifact.get("adapter"), dict) else {}
        gguf = artifact.get("gguf") if isinstance(artifact.get("gguf"), dict) else {}
        ollama = artifact.get("ollama") if isinstance(artifact.get("ollama"), dict) else {}
        run = (
            artifact.get("trainingRun")
            if isinstance(artifact.get("trainingRun"), dict)
            else {}
        )
        summary = {
            "alias": alias,
            "lineageId": _string(artifact.get("lineageId")),
            "version": _string(artifact.get("version")),
            "role": _string(artifact.get("role")),
            "trainingLabel": _string(artifact.get("label")),
            "includeInControlledClaims": _bool(artifact.get("includeInControlledClaims")),
            "expectedTag": _string(ollama.get("tag")),
            "ollamaDigest": _string(ollama.get("digest")),
            "baseQuantization": _string(ollama.get("baseQuantization")),
            "huggingFaceId": None,
            "adapterSha256": _hex_digest(adapter.get("sha256")),
            "adapterDtype": _string(adapter.get("safetensorsDtype")),
            "ggufBlobSha256": _hex_digest(ollama.get("adapterBlob")),
            "ggufDtype": _string(gguf.get("tensorDtype")),
            "ggufBytes": _int(gguf.get("bytes")),
            "gitCommitSha": _string(run.get("gitCommitSha")),
        }

    if summary["expectedTag"] is None:
        raise LineageUnavailable(
            f'The lineage record names no Ollama tag for alias "{alias}".'
        )
    assert tuple(summary) == LINEAGE_SUMMARY_FIELDS
    return summary
