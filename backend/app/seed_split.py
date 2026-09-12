"""Deterministic, mixed-format train/validation split for one course's approved seeds.

Reads data/exports/{courseId}/approved-finetune.jsonl (validated on export) and
writes train.jsonl, validation.jsonl and manifest.json beside it.

The mixed objective (v4)
------------------------
A fine-tuned model is used in two conditions: alone, answering a bare question
with no context, and grounded, answering the shared grounded prompt built
from retrieved syllabus excerpts. One training format cannot serve both. The
split therefore renders two kinds of record from the approved seeds:

- **bare**: `instruction` is the seed's question, `response` its answer. The
  normal Fine-Tuned input format, so standalone Fine-Tuned stays a meaningful
  condition. Only answerable seeds become bare records.
- **grounded**: `instruction` is the production grounded prompt,
  `grounded_generation.build_grounded_prompt`, rendered over the excerpts the
  production retriever returns for the seed's question; `response` is the
  seed's answer. Every seed becomes one grounded record: answerable seeds
  whose answer the retrieved excerpts support, and the authored behaviour
  seeds, whose `questionType` says whether they teach abstention
  (`unanswerable`) or false-premise correction (`false_premise`).

`instruction` is always the user turn, so the trainer, the launch-time
validator and the worker transfer read every record the same way; `format`,
`kind`, `question` and `context` are metadata. An answerable seed whose
numbers the retrieved excerpts do not contain is not rendered grounded: it
would teach the model to state a fact its context does not hold, which is the
memorisation this format exists to avoid. The same check applies to the
behaviour seeds: an abstention's "what the syllabus does say" and a
correction's "what the rule actually is" cite figures, and a figure the
retriever did not surface for that question cannot be taught from that
prompt. Every seed not rendered grounded is listed in the manifest with the
figures it was missing, so the author can rephrase it or improve retrieval.

The split is stratified by `format/kind` so validation holds every kind, and
the manifest records the realised composition against the intended targets,
the prompt template's fingerprint, and the index the excerpts came from.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import HTTPException

from app.grounded_generation import (
    PROMPT_TEMPLATE_NAME,
    build_grounded_prompt,
    prompt_template_fingerprint,
)
from app.retrieval_diversity import DEFAULT_TOP_K
from app.retrieval_facets import extract_question_facets
from app.seed_export import (
    FinetuneJsonlValidationError,
    course_export_dir,
    validate_finetune_jsonl,
    write_json,
    write_jsonl,
)
from app.storage import CourseArtifactStorage, get_course_artifact_storage

DEFAULT_SPLIT_SEED = 360
DEFAULT_VALIDATION_FRACTION = 0.1
SOURCE_FILENAME = "approved-finetune.jsonl"
TRAIN_FILENAME = "train.jsonl"
VALIDATION_FILENAME = "validation.jsonl"
MANIFEST_FILENAME = "manifest.json"
SUMMARY_FILENAME = "approved-export-summary.json"

DATASET_FORMAT = "mixed-v4"

BARE = "bare"
GROUNDED = "grounded"
ANSWERABLE = "answerable"
ABSTAIN = "abstain"
FALSE_PREMISE = "false_premise"

#: A seed's `questionType` decides its kind. Anything else, including the
#: generator's `direct`, is an answerable seed.
BEHAVIOUR_QUESTION_TYPES: dict[str, str] = {
    "unanswerable": ABSTAIN,
    "false_premise": FALSE_PREMISE,
}

GROUP_KEYS = (
    f"{BARE}/{ANSWERABLE}",
    f"{GROUNDED}/{ANSWERABLE}",
    f"{GROUNDED}/{ABSTAIN}",
    f"{GROUNDED}/{FALSE_PREMISE}",
)

#: The intended mix, recorded beside the realised one. The realised mix is
#: data-driven: every reviewed seed is used, and the manifest says how far
#: the result is from these.
COMPOSITION_TARGETS: dict[str, float] = {
    f"{BARE}/{ANSWERABLE}": 0.40,
    f"{GROUNDED}/{ANSWERABLE}": 0.35,
    f"{GROUNDED}/{ABSTAIN}": 0.15,
    f"{GROUNDED}/{FALSE_PREMISE}": 0.10,
}

CHECKSUM_CHUNK_BYTES = 64 * 1024

Retriever = Callable[..., Awaitable[tuple[str, list[dict[str, Any]]]]]


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHECKSUM_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class TrainingSplitError(ValueError):
    """Raised when a training split cannot be prepared from the approved export."""


def approved_finetune_path(course_id: str, *, export_root: Path | None = None) -> Path:
    return course_export_dir(course_id, root=export_root) / SOURCE_FILENAME


def compute_validation_size(total: int, *, fraction: float = DEFAULT_VALIDATION_FRACTION) -> int:
    """Return validation size (~fraction), leaving at least one train example."""
    if total < 2:
        raise TrainingSplitError(
            "Need at least 2 approved examples to create a train/validation split."
        )
    validation_count = max(1, math.ceil(total * fraction))
    validation_count = min(validation_count, total - 1)
    return validation_count


def load_approved_finetune_records(
    path: Path,
) -> list[dict[str, Any]]:
    """Validate and load approved-finetune.jsonl.

    `instruction` and `response` are required; `questionType`,
    `sourceChunkIds` and `seedId` are kept when the export wrote them.
    """
    if not path.is_file():
        raise TrainingSplitError(
            f'Approved export file does not exist: {path}. '
            "Export approved seeds before preparing a training split."
        )

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    try:
        validate_finetune_jsonl(path, expected_count=len(lines))
    except FinetuneJsonlValidationError as exc:
        raise TrainingSplitError(str(exc)) from exc

    records: list[dict[str, Any]] = []
    for line in lines:
        payload = json.loads(line)
        record: dict[str, Any] = {
            "instruction": payload["instruction"],
            "response": payload["response"],
        }
        question_type = payload.get("questionType")
        if isinstance(question_type, str) and question_type.strip():
            record["questionType"] = question_type.strip().lower()
        chunk_ids = payload.get("sourceChunkIds")
        if isinstance(chunk_ids, list):
            record["sourceChunkIds"] = [str(item) for item in chunk_ids if str(item).strip()]
        seed_id = payload.get("seedId")
        if isinstance(seed_id, str) and seed_id.strip():
            record["seedId"] = seed_id.strip()
        records.append(record)
    return records


def read_source_export_timestamp(
    course_id: str,
    *,
    export_root: Path | None = None,
) -> str | None:
    summary_path = course_export_dir(course_id, root=export_root) / SUMMARY_FILENAME
    if not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    exported_at = payload.get("exportedAt")
    if isinstance(exported_at, str) and exported_at.strip():
        return exported_at.strip()
    return None


# --------------------------------------------------------------------------- #
# Kinds and support
# --------------------------------------------------------------------------- #


def record_kind(record: dict[str, Any]) -> str:
    question_type = str(record.get("questionType") or "").strip().lower()
    return BEHAVIOUR_QUESTION_TYPES.get(question_type, ANSWERABLE)


def group_key(record: dict[str, Any]) -> str:
    return f"{record.get('format', BARE)}/{record.get('kind', ANSWERABLE)}"


#: A number as a student reads one: 48, 11:59, 97.0%, 800-1200, 1-7, 12/14.
_NUMBER_PATTERN = re.compile(r"\d+(?:[.,:/-]\d+)*%?")


def numeric_tokens(text: str) -> list[str]:
    """The numbers in `text`, in order, without duplicates."""
    seen: list[str] = []
    for token in _NUMBER_PATTERN.findall(text or ""):
        if token not in seen:
            seen.append(token)
    return seen


def _contains_number(haystack: str, token: str) -> bool:
    """`token` as a whole number: `1` is not found inside `11:59` or `15%`,
    but `48` is found in `48-hour` and `3` in `Sprint 3:`."""
    pattern = rf"(?<![\d.,:/-]){re.escape(token)}(?!\d|[.,:/-]\d)"
    return re.search(pattern, haystack) is not None


def unsupported_numbers(response: str, context_text: str) -> list[str]:
    """Numbers the answer states that the excerpts do not contain.

    A number is the smallest claim a syllabus answer makes, and the one a
    model invents most readily. An answer whose every number appears in its
    excerpts is not proven correct by this, but an answer with a number the
    excerpts lack is a grounded example the excerpts cannot teach.
    """
    haystack = " ".join((context_text or "").split())
    return [token for token in numeric_tokens(response) if not _contains_number(haystack, token)]


#: Words that say nothing about which excerpt an answer came from. The
#: overlap measures in `evaluation/check_overlap.py` use a similar list; this
#: one adds the words every syllabus excerpt shares.
_SUPPORT_STOPWORDS = frozenset(
    {
        "about", "after", "all", "also", "and", "any", "are", "before", "but", "can",
        "class", "classes", "course", "does", "during", "each", "for", "from", "get",
        "has", "have", "how", "into", "instructor", "its", "many", "much", "must",
        "need", "not", "one", "only", "other", "our", "per", "professor", "should",
        "student", "students", "syllabus", "than", "that", "the", "their", "them",
        "then", "there", "these", "they", "this", "two", "use", "used", "using", "via",
        "was", "what", "when", "where", "which", "who", "will", "with", "yes", "you",
        "your",
    }
)
_WORD_PATTERN = re.compile(r"[a-z0-9]+")

#: A chunk that shares this fraction of an answer's content words supports it
#: on its own; below it, only a matching figure can implicate a chunk.
LEXICAL_SUPPORT_THRESHOLD = 0.4
#: The least overlap that lets a figure match or a best-effort attribution
#: count. An answer whose best excerpt shares less than this of its wording
#: is not an answer those excerpts can teach.
LEXICAL_SUPPORT_FLOOR = 0.2


def _stem(word: str) -> str:
    """Crude suffix stripping so `results`/`result`, `assesses`/`assess` and
    `described`/`describes` meet. Plural first, then a verb ending, then a
    trailing `e`, so `recordings`, `recording`, `recorded` and `record` agree."""
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def content_stems(text: str) -> set[str]:
    return {
        _stem(word)
        for word in _WORD_PATTERN.findall((text or "").lower())
        if len(word) >= 3 and word not in _SUPPORT_STOPWORDS
    }


def containment(answer_stems: set[str], text: str) -> float:
    """Share of the answer's content words that `text` contains."""
    if not answer_stems:
        return 0.0
    return len(answer_stems & content_stems(text)) / len(answer_stems)


def support_analysis(response: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Which retrieved excerpts support an answer, and whether any does.

    Two signals, because neither alone attributes well. Figures are exact but
    shared (every deadline chunk says 11:59 p.m.), so a figure implicates a
    chunk only when the chunk holds every figure the answer states, or,
    failing that, one of them together with some of the answer's wording.
    Wording is fuzzy but specific: an answer without figures (the first
    plagiarism incident is a zero on the assignment) is attributed to the
    excerpt that shares most of its content words. `supported` is false when
    no excerpt qualifies, which is what stops an answer the excerpts do not
    hold from becoming a grounded example.
    """
    numbers = numeric_tokens(response)
    stems = content_stems(response)
    haystack = " ".join("\n".join(chunk["text"] for chunk in chunks).split())
    missing = [token for token in numbers if not _contains_number(haystack, token)]

    scored = []
    for chunk in chunks:
        text = " ".join(chunk["text"].split())
        hits = sum(1 for token in numbers if _contains_number(text, token))
        scored.append((chunk["chunk_id"], containment(stems, text), hits))

    support: list[str] = []
    if numbers:
        support = [chunk_id for chunk_id, _, hits in scored if hits == len(numbers)]
        if not support:
            support = [
                chunk_id
                for chunk_id, overlap, hits in scored
                if hits > 0 and overlap >= LEXICAL_SUPPORT_FLOOR
            ]
    if not support:
        support = [
            chunk_id for chunk_id, overlap, _ in scored if overlap >= LEXICAL_SUPPORT_THRESHOLD
        ]
    if not support and scored:
        best_id, best_overlap, _ = max(scored, key=lambda item: item[1])
        if best_overlap >= LEXICAL_SUPPORT_FLOOR:
            support = [best_id]

    return {
        "numbers": numbers,
        "missingNumbers": missing,
        "supportChunkIds": support,
        "bestContainment": round(max((overlap for _, overlap, _ in scored), default=0.0), 3),
        "supported": not missing and bool(support),
    }


#: An abstention's first sentence says what the syllabus does not say; any
#: sentence after it says what the syllabus does say, and that part is a
#: citation like any other. This is the share of its content words the
#: retrieved excerpts must hold.
ABSTENTION_CITATION_COVERAGE = 0.5
#: Below this many content words a trailing sentence is a turn of phrase, not
#: a citation, and is not checked.
ABSTENTION_CITATION_MIN_WORDS = 3


def abstention_citation_analysis(
    question: str, response: str, chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    """Whether what an abstention says the syllabus *does* say is in the excerpts.

    The abstaining sentence restates the question and cannot be supported by
    anything; the sentences after it are checked as citations, with the
    question's own words set aside so that restating the topic is not counted
    against the record.
    """
    first, separator, rest = response.strip().partition(". ")
    cited = content_stems(rest if separator else "") - content_stems(question)
    if len(cited) < ABSTENTION_CITATION_MIN_WORDS:
        return {"cited": sorted(cited), "coverage": None, "supported": True}
    union = content_stems("\n".join(chunk["text"] for chunk in chunks))
    coverage = len(cited & union) / len(cited)
    return {
        "cited": sorted(cited),
        "coverage": round(coverage, 3),
        "supported": coverage >= ABSTENTION_CITATION_COVERAGE,
        "missing": sorted(cited - union),
    }


def _context_entries(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunkId": chunk["chunk_id"],
            "section": chunk["section"],
            "text": chunk["text"],
        }
        for chunk in chunks
    ]


# --------------------------------------------------------------------------- #
# Building the mixed dataset
# --------------------------------------------------------------------------- #


async def build_mixed_records(
    course_id: str,
    records: list[dict[str, Any]],
    *,
    retrieve: Retriever,
    storage: CourseArtifactStorage | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every approved seed as its bare and grounded records.

    `retrieve` is the production retriever (`retrieve_course_syllabus_chunks`);
    a test passes a fake. Returns the records and the list of answerable seeds
    that were not rendered grounded, each with the reason.
    """
    bare: list[dict[str, Any]] = []
    grounded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        seed_id = record.get("seedId") or f"record-{index}"
        kind = record_kind(record)
        question = " ".join(str(record["instruction"]).split())
        response = str(record["response"]).strip()

        if kind == ANSWERABLE:
            bare.append(
                {
                    "format": BARE,
                    "kind": ANSWERABLE,
                    "seedId": seed_id,
                    "question": question,
                    "instruction": question,
                    "response": response,
                }
            )

        try:
            _, chunks = await retrieve(
                course_id=course_id, question=question, top_k=top_k, storage=storage
            )
        except HTTPException as exc:
            raise TrainingSplitError(
                f"Retrieval failed for seed {seed_id}: {exc.detail}"
            ) from exc
        if not chunks:
            skipped.append(
                {"seedId": seed_id, "kind": kind, "question": question, "reason": "no excerpts retrieved"}
            )
            continue

        # Every grounded record, whatever its kind: an abstention that says what
        # the syllabus does say, and a correction that gives the real rule, cite
        # figures too, and those must come from the excerpts in the prompt.
        analysis = support_analysis(response, chunks)
        if analysis["missingNumbers"]:
            skipped.append(
                {
                    "seedId": seed_id,
                    "kind": kind,
                    "question": question,
                    "reason": "response states numbers the retrieved excerpts do not contain",
                    "missing": analysis["missingNumbers"],
                }
            )
            continue
        # An answer and a correction must come from some excerpt in the prompt.
        # An abstention's claim is an absence, which no excerpt can support;
        # what it goes on to say the syllabus does say is checked as a citation.
        if kind != ABSTAIN and not analysis["supportChunkIds"]:
            skipped.append(
                {
                    "seedId": seed_id,
                    "kind": kind,
                    "question": question,
                    "reason": "no retrieved excerpt supports the response",
                    "bestContainment": analysis["bestContainment"],
                }
            )
            continue
        if kind == ABSTAIN:
            citation = abstention_citation_analysis(question, response, chunks)
            if not citation["supported"]:
                skipped.append(
                    {
                        "seedId": seed_id,
                        "kind": kind,
                        "question": question,
                        "reason": "abstention cites details the retrieved excerpts do not contain",
                        "missing": citation["missing"],
                        "coverage": citation["coverage"],
                    }
                )
                continue

        facets = extract_question_facets(question)
        grounded.append(
            {
                "format": GROUNDED,
                "kind": kind,
                "seedId": seed_id,
                "question": question,
                "context": _context_entries(chunks),
                "supportChunkIds": [] if kind == ABSTAIN else analysis["supportChunkIds"],
                "instruction": build_grounded_prompt(question, chunks, facets),
                "response": response,
            }
        )

    return bare + grounded, skipped


def split_records(
    records: list[dict[str, Any]],
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically shuffle then split one group into train and validation."""
    total = len(records)
    validation_count = compute_validation_size(total, fraction=validation_fraction)
    shuffled = list(records)
    rng = random.Random(split_seed)
    rng.shuffle(shuffled)
    validation = shuffled[:validation_count]
    train = shuffled[validation_count:]
    if len(train) + len(validation) != total:
        raise TrainingSplitError(
            "Split counts do not add up to the source count "
            f"({len(train)} train + {len(validation)} validation != {total})."
        )
    return train, validation


def split_mixed_records(
    records: list[dict[str, Any]],
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified by `format/kind`: validation holds every kind that has two or
    more examples; a lone example trains. Deterministic for a seed."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(group_key(record), []).append(record)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for members in groups.values():
        if len(members) < 2:
            train.extend(members)
            continue
        group_train, group_validation = split_records(
            members, split_seed=split_seed, validation_fraction=validation_fraction
        )
        train.extend(group_train)
        validation.extend(group_validation)

    random.Random(split_seed).shuffle(train)
    random.Random(split_seed).shuffle(validation)
    if len(train) + len(validation) != len(records):
        raise TrainingSplitError(
            "Split counts do not add up to the source count "
            f"({len(train)} train + {len(validation)} validation != {len(records)})."
        )
    return train, validation


def composition(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {key: 0 for key in GROUP_KEYS}
    for record in records:
        key = group_key(record)
        counts[key] = counts.get(key, 0) + 1
    return counts


def composition_percentages(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {key: 0.0 for key in counts}
    return {key: round(value / total, 3) for key, value in counts.items()}


def approved_export_status(
    course_id: str,
    *,
    export_root: Path | None = None,
) -> dict[str, Any]:
    """Return whether an approved-finetune.jsonl export exists for the course."""
    path = approved_finetune_path(course_id, export_root=export_root)
    exists = path.is_file()
    example_count = 0
    if exists:
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        example_count = len(lines)
    return {
        "courseId": course_id,
        "exists": exists,
        "exportPath": str(path),
        "exampleCount": example_count,
        "sourceFile": SOURCE_FILENAME,
    }


async def _production_retriever(**kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
    from app.course_rag import retrieve_course_syllabus_chunks

    return await retrieve_course_syllabus_chunks(**kwargs)


def _index_description(storage: CourseArtifactStorage, course_id: str) -> dict[str, Any]:
    path = storage.index_path(course_id)
    if not path.is_file():
        return {"indexSha256": None, "indexChunkCount": None}
    index = storage.load_index(course_id) or {}
    chunks = index.get("chunks")
    return {
        "indexSha256": sha256_file(path),
        "indexChunkCount": len(chunks) if isinstance(chunks, list) else None,
    }


async def prepare_training_split(
    course_id: str,
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    export_root: Path | None = None,
    created_at: str | None = None,
    storage: CourseArtifactStorage | None = None,
    retrieve: Retriever | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Create train.jsonl, validation.jsonl and manifest.json from the approved export.

    Asynchronous because rendering the grounded records runs the production
    retriever, which embeds each question through Ollama.
    """
    out_dir = course_export_dir(course_id, root=export_root)
    source_path = out_dir / SOURCE_FILENAME
    records = load_approved_finetune_records(source_path)
    if len(records) < 2:
        raise TrainingSplitError(
            "Need at least 2 approved examples to create a train/validation split."
        )

    artifact_storage = storage or get_course_artifact_storage()
    mixed, skipped = await build_mixed_records(
        course_id,
        records,
        retrieve=retrieve or _production_retriever,
        storage=artifact_storage,
        top_k=top_k,
    )
    train, validation = split_mixed_records(
        mixed,
        split_seed=split_seed,
        validation_fraction=validation_fraction,
    )

    train_path = out_dir / TRAIN_FILENAME
    validation_path = out_dir / VALIDATION_FILENAME
    manifest_path = out_dir / MANIFEST_FILENAME

    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)

    if len(train) + len(validation) != len(mixed):
        raise TrainingSplitError(
            "Output counts do not add up to the source count "
            f"({len(train)} + {len(validation)} != {len(mixed)})."
        )

    created = created_at or datetime.now(timezone.utc).isoformat()
    # Written into the manifest so the split is self-describing wherever it ends
    # up. The worker verifies each transferred file against these before letting
    # it replace anything on the cluster.
    checksums = {
        TRAIN_FILENAME: sha256_file(train_path),
        VALIDATION_FILENAME: sha256_file(validation_path),
    }
    source_export_timestamp = read_source_export_timestamp(
        course_id,
        export_root=export_root,
    )
    dataset_version = f"{course_id}-mixed-split-seed{split_seed}-n{len(mixed)}"
    total_counts = composition(mixed)

    manifest = {
        "courseId": course_id,
        "format": DATASET_FORMAT,
        "datasetVersion": dataset_version,
        "sourceFile": str(source_path),
        "sourceExportTimestamp": source_export_timestamp,
        "createdAt": created,
        "splitSeed": split_seed,
        "sourceExamples": len(records),
        "totalExamples": len(mixed),
        "trainExamples": len(train),
        "validationExamples": len(validation),
        "trainFile": str(train_path),
        "validationFile": str(validation_path),
        "composition": {
            "train": composition(train),
            "validation": composition(validation),
            "total": total_counts,
            "percentages": composition_percentages(total_counts),
        },
        "compositionTargets": dict(COMPOSITION_TARGETS),
        "groundedSkipped": skipped,
        "promptTemplate": {
            "name": PROMPT_TEMPLATE_NAME,
            "sha256": prompt_template_fingerprint(),
        },
        "retrieval": {"topK": top_k, **_index_description(artifact_storage, course_id)},
        "checksums": checksums,
        "checksumAlgorithm": "sha256",
    }
    write_json(manifest_path, manifest)

    return {
        "courseId": course_id,
        "manifest": manifest,
        "trainExamples": len(train),
        "validationExamples": len(validation),
        "totalExamples": len(mixed),
        "sourceExamples": len(records),
        "composition": manifest["composition"],
        "groundedSkipped": skipped,
        "splitSeed": split_seed,
        "files": {
            "trainJsonl": str(train_path),
            "validationJsonl": str(validation_path),
            "manifestJson": str(manifest_path),
            "sourceJsonl": str(source_path),
        },
    }
