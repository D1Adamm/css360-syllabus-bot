"""Rebuild per-course RAG indexes from existing extracted syllabus.txt files.

Usage:
  python -m app.reindex_course --course-id css-360-winter-2026-a7rp
  python -m app.reindex_course --all
  python -m app.reindex_course --course-id css-360-winter-2026-a7rp --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from app.course_id import assert_valid_course_id
from app.course_index import build_course_rag_index
from app.rag import OLLAMA_EMBEDDING_MODEL
from app.storage import CourseArtifactStorage, LocalCourseArtifactStorage, get_course_artifact_storage
from app.syllabus_chunking import (
    INDEX_VERSION,
    chunk_syllabus_text,
    summarize_chunking,
    validate_chunks,
)


def _list_local_course_ids(storage: LocalCourseArtifactStorage) -> list[str]:
    if not storage.root_dir.is_dir():
        return []
    course_ids: list[str] = []
    for path in sorted(storage.root_dir.iterdir()):
        if not path.is_dir():
            continue
        syllabus = path / "syllabus.txt"
        if syllabus.is_file():
            course_ids.append(path.name)
    return course_ids


def _print_dry_run(course_id: str, syllabus_text: str) -> dict[str, Any]:
    chunks = chunk_syllabus_text(syllabus_text)
    summary = summarize_chunking(chunks)
    warnings = validate_chunks(
        chunks,
        document_title=summary.get("documentTitle", ""),
        source_char_count=len(syllabus_text),
        strict=False,
    )
    print(f"[dry-run] courseId={course_id}")
    print(f"  indexVersion={INDEX_VERSION}")
    print(f"  embeddingModel={OLLAMA_EMBEDDING_MODEL}")
    print(f"  documentTitle={summary.get('documentTitle')!r}")
    print(f"  sectionCount={summary.get('sectionCount')}")
    print(f"  chunkCount={summary.get('chunkCount')}")
    print("  detectedHeadings:")
    for title in summary.get("sectionTitles", []):
        count = sum(1 for chunk in chunks if chunk.section_title == title)
        print(f"    - {title} ({count} chunk{'s' if count != 1 else ''})")
    if warnings:
        print("  validationWarnings:")
        for warning in warnings:
            print(f"    - {warning}")
    else:
        print("  validationWarnings: none")
    return {"courseId": course_id, "dryRun": True, **summary, "warnings": warnings}


async def _reindex_one(
    *,
    course_id: str,
    storage: CourseArtifactStorage,
    dry_run: bool,
    strict: bool,
) -> dict[str, Any]:
    safe_id = assert_valid_course_id(course_id)
    text = storage.load_extracted_text(safe_id)
    if text is None or not text.strip():
        raise FileNotFoundError(
            f"No extracted syllabus.txt found for courseId={safe_id} "
            f"(expected under course data directory)."
        )

    source_path = storage.extracted_text_path(safe_id)
    if dry_run:
        return _print_dry_run(safe_id, text)

    index_data = await build_course_rag_index(
        course_id=safe_id,
        source_file=str(source_path.name),
        syllabus_text=text,
        storage=storage,
        strict_validation=strict,
    )
    summary = {
        "courseId": safe_id,
        "indexVersion": index_data.get("indexVersion", INDEX_VERSION),
        "documentTitle": index_data.get("documentTitle", ""),
        "chunkCount": index_data.get("chunkCount", 0),
        "sectionCount": index_data.get("sectionCount", 0),
        "indexPath": str(storage.index_path(safe_id)),
        "sourcePath": str(source_path),
        "embeddingModel": index_data.get("embeddingModel", OLLAMA_EMBEDDING_MODEL),
        "warnings": index_data.get("chunkingWarnings", []),
    }
    print(f"[reindexed] courseId={safe_id}")
    print(f"  indexVersion={summary['indexVersion']}")
    print(f"  documentTitle={summary['documentTitle']!r}")
    print(f"  sectionCount={summary['sectionCount']}")
    print(f"  chunkCount={summary['chunkCount']}")
    print(f"  embeddingModel={summary['embeddingModel']}")
    print(f"  wrote={summary['indexPath']}")
    print(f"  preservedSyllabus={summary['sourcePath']}")
    print("  fact inventory cache invalidated via atomic index write")
    if summary["warnings"]:
        print("  validationWarnings:")
        for warning in summary["warnings"]:
            print(f"    - {warning}")
    return summary


async def _async_main(args: argparse.Namespace) -> int:
    storage = get_course_artifact_storage()
    if not isinstance(storage, LocalCourseArtifactStorage):
        print("Only local course storage is supported for reindexing.", file=sys.stderr)
        return 2

    if args.all:
        course_ids = _list_local_course_ids(storage)
        if not course_ids:
            print(f"No courses with syllabus.txt found under {storage.root_dir}")
            return 1
    else:
        course_ids = [args.course_id]

    failures = 0
    for course_id in course_ids:
        try:
            await _reindex_one(
                course_id=course_id,
                storage=storage,
                dry_run=args.dry_run,
                strict=args.strict,
            )
        except Exception as exc:  # noqa: BLE001 - CLI reports and continues
            failures += 1
            print(f"[error] courseId={course_id}: {exc}", file=sys.stderr)
            if not args.all:
                return 1

    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild course RAG indexes with heading-aware chunking.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--course-id",
        dest="course_id",
        help="Rebuild a single course index from its syllabus.txt",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Rebuild indexes for every local course that has syllabus.txt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show proposed sections/chunks without writing indexes or calling Ollama",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when chunking validation warnings are produced",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
