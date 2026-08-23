#!/usr/bin/env python3
"""Compare qwen3:4b vs qwen3:8b on the same 10 allocated facts.

Does not save seeds or change SEED_GENERATION_MODEL.

Usage (from repo root):

  PYTHONPATH=backend backend/.venv/bin/python \\
    backend/scripts/benchmark_seed_models.py \\
    --course-id css-360-winter-2026-a7rp

Outputs:
  /tmp/qwen3-4b-benchmark.json
  /tmp/qwen3-8b-benchmark.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.seed_model_benchmark import (  # noqa: E402
    DEFAULT_BENCHMARK_COURSE_ID,
    DEFAULT_BENCHMARK_FACT_COUNT,
    DEFAULT_MODELS,
    compare_benchmark_summaries,
    format_comparison_report,
    load_benchmark_context,
    run_model_benchmark,
    write_benchmark_json,
)

DEFAULT_OUTPUTS = {
    "qwen3:4b": Path("/tmp/qwen3-4b-benchmark.json"),
    "qwen3:8b": Path("/tmp/qwen3-8b-benchmark.json"),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline seed-generation model benchmark "
            "(same facts, nothing persisted)."
        )
    )
    parser.add_argument(
        "--course-id",
        default=DEFAULT_BENCHMARK_COURSE_ID,
        help=f"Course id (default: {DEFAULT_BENCHMARK_COURSE_ID})",
    )
    parser.add_argument(
        "--fact-count",
        type=int,
        default=DEFAULT_BENCHMARK_FACT_COUNT,
        help=f"Number of allocated facts (default: {DEFAULT_BENCHMARK_FACT_COUNT})",
    )
    parser.add_argument(
        "--force-refresh-inventory",
        action="store_true",
        help="Rebuild fact inventory instead of using the disk cache.",
    )
    parser.add_argument(
        "--4b-output",
        dest="output_4b",
        type=Path,
        default=DEFAULT_OUTPUTS["qwen3:4b"],
        help=f"Output path for 4b results (default: {DEFAULT_OUTPUTS['qwen3:4b']})",
    )
    parser.add_argument(
        "--8b-output",
        dest="output_8b",
        type=Path,
        default=DEFAULT_OUTPUTS["qwen3:8b"],
        help=f"Output path for 8b results (default: {DEFAULT_OUTPUTS['qwen3:8b']})",
    )
    return parser.parse_args(argv)


async def _async_main(args: argparse.Namespace) -> int:
    print(
        f"Loading benchmark context for {args.course_id} "
        f"(fact_count={args.fact_count})...",
        flush=True,
    )
    context = await load_benchmark_context(
        course_id=args.course_id,
        fact_count=args.fact_count,
        force_refresh=args.force_refresh_inventory,
    )
    selected = context["selected"]
    chunk_lookup = context["chunkLookup"]
    print(f"Selected facts: {', '.join(context['factIds'])}", flush=True)

    results: dict[str, dict] = {}
    outputs = {
        "qwen3:4b": args.output_4b,
        "qwen3:8b": args.output_8b,
    }

    for model in DEFAULT_MODELS:
        print(f"\n=== Running {model} ===", flush=True)
        payload = await run_model_benchmark(
            model=model,
            selected=selected,
            chunk_lookup=chunk_lookup,
        )
        payload["courseId"] = context["courseId"]
        path = write_benchmark_json(outputs[model], payload)
        results[model] = payload
        summary = payload["summary"]
        print(
            f"Wrote {path} "
            f"(accepted={summary['acceptedCount']}/"
            f"{summary['candidateCount']})",
            flush=True,
        )

    left = results["qwen3:4b"]
    right = results["qwen3:8b"]
    print("\n=== Comparison ===", flush=True)
    print(format_comparison_report(course_id=args.course_id, left=left, right=right))
    # Keep structured comparison available for tooling.
    _ = compare_benchmark_summaries(left, right)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
