#!/usr/bin/env python3
"""Check one saved response from the CSS 360 benchmark smoke, condition by condition.

    python3 backend/scripts/check_css360_benchmark_smoke.py /tmp/smoke-ft_rag_v2.json --condition ft_rag:v2

Standard library only, so it runs with the VM's system `python3` on a file
curl saved, from either the loopback request or the one through Nginx. It
prints one PASS or FAIL line per check, then the served tag, the served
digest, the timing and the answer, and exits 0 only if every check passed.

What is checked: the route and the fixed course; the prompt template
fingerprint and the retrieval block (pair only); the decoding options; that
exactly the expected condition came back; that it is `status: ok` with no
error and outcome `scorable`; its three integrity flags, `tagMatchesLineage`,
`decodingMatchesSpec` and `promptEchoMatches`; that the served tag is the
tag the lineage record names; that an Ollama digest was recorded and matches
the record; that the answer has text; that the timing is present; and that
the summary counts the one attempt.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

COURSE_ID = "css-360-winter-2026-a7rp"
#: `grounded_generation.prompt_template_fingerprint()` at the deployed commit.
#: A backend test pins this constant to the live value.
EXPECTED_TEMPLATE_SHA256 = "c963ffb5bd97aa946e730fb916b7ddedcf3a7cab9049199be186b994752d7a93"
#: `grounded_generation.grounded_options()`; also pinned by a backend test.
EXPECTED_DECODING: dict[str, Any] = {
    "num_predict": 256,
    "temperature": 0,
    "repeat_penalty": 1.05,
    "repeat_last_n": 4096,
    "seed": 360,
    "num_ctx": 4096,
}
INTEGRITY_FLAGS = ("tagMatchesLineage", "decodingMatchesSpec", "promptEchoMatches")
HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")

Check = tuple[str, bool, str]


def route_for(condition: str) -> str:
    return "pair" if condition == "rag" or condition.startswith("ft_rag:") else "standalone"


def _get(payload: Any, *keys: str) -> Any:
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def check_response(payload: Any, *, condition: str) -> list[Check]:
    """Every check for one expected condition, in the order they are printed."""
    checks: list[Check] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    if not isinstance(payload, dict):
        add("response is a JSON object", False, type(payload).__name__)
        return checks

    route = route_for(condition)
    add("route and course",
        payload.get("route") == route and payload.get("courseId") == COURSE_ID,
        f"route={payload.get('route')!r} courseId={payload.get('courseId')!r}")

    if route == "pair":
        template = payload.get("promptTemplate") or {}
        add("prompt template",
            template.get("name") == "grounded-v1" and template.get("sha256") == EXPECTED_TEMPLATE_SHA256,
            f"{template!r}")
        retrieval = payload.get("retrieval") or {}
        chunks = payload.get("retrievedChunks") or []
        add("retrieval",
            retrieval.get("topK") == 4 and isinstance(retrieval.get("chunkCount"), int)
            and retrieval.get("chunkCount", 0) >= 1 and len(chunks) == retrieval.get("chunkCount")
            and isinstance(retrieval.get("setSha256"), str) and len(retrieval.get("setSha256", "")) == 64,
            f"topK={retrieval.get('topK')!r} chunkCount={retrieval.get('chunkCount')!r}")
    else:
        add("no retrieval",
            payload.get("retrieval") is None and payload.get("retrievedChunks") == []
            and payload.get("promptTemplate") is None and payload.get("prompt") == payload.get("question"),
            "standalone sends the bare question")

    add("prompt hash",
        isinstance(payload.get("promptSha256"), str) and HEX_DIGEST.match(payload.get("promptSha256") or "") is not None,
        f"{payload.get('promptSha256')!r}")
    add("decoding", payload.get("decoding") == EXPECTED_DECODING, f"{payload.get('decoding')!r}")

    conditions = payload.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != 1:
        add("exactly one condition", False, f"{len(conditions) if isinstance(conditions, list) else conditions!r}")
        return checks
    c = conditions[0]
    if not isinstance(c, dict):
        add("exactly one condition", False, type(c).__name__)
        return checks
    add("exactly one condition", c.get("condition") == condition, f"{c.get('condition')!r}")
    add("status ok, no error, scorable",
        c.get("status") == "ok" and c.get("error") is None and c.get("outcome") == "scorable",
        f"status={c.get('status')!r} outcome={c.get('outcome')!r} error={c.get('error')!r}")
    for flag in INTEGRITY_FLAGS:
        add(f"flag {flag}", c.get(flag) is True, f"{c.get(flag)!r}")
    expected_tag = _get(c, "lineage", "expectedTag")
    add("served tag is the lineage tag",
        isinstance(expected_tag, str) and c.get("servedTag") == expected_tag,
        f"servedTag={c.get('servedTag')!r} expectedTag={expected_tag!r}")
    served_digest = c.get("servedDigest")
    add("digest recorded",
        isinstance(served_digest, str) and HEX_DIGEST.match(served_digest) is not None,
        f"{served_digest!r}")
    add("digest matches lineage", c.get("digestMatchesLineage") is True,
        f"digestMatchesLineage={c.get('digestMatchesLineage')!r} "
        f"lineage.ollamaDigest={_get(c, 'lineage', 'ollamaDigest')!r}")
    add("answer has text", isinstance(c.get("answer"), str) and bool(c.get("answer", "").strip()),
        f"{len(c.get('answer') or '')} characters")
    timing = c.get("timing") or {}
    add("timing recorded",
        isinstance(timing.get("wallSeconds"), (int, float))
        and isinstance(timing.get("generationSeconds"), (int, float))
        and _get(timing, "ollama", "doneReason") in ("stop", "length"),
        f"{timing!r}")
    summary = payload.get("summary") or {}
    add("summary counts one attempt",
        summary.get("attempted") == 1 and summary.get("scorable") == 1,
        f"{summary!r}")
    return checks


def report(payload: Any, checks: list[Check]) -> str:
    lines = [f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f"  ({detail})") for name, ok, detail in checks]
    condition = (payload.get("conditions") or [{}])[0] if isinstance(payload, dict) else {}
    if isinstance(condition, dict):
        lines.append(f"servedTag: {condition.get('servedTag')!r}")
        lines.append(f"servedDigest: {condition.get('servedDigest')!r}")
        lines.append(f"timing: {json.dumps(condition.get('timing'))}")
        lines.append(f"answer: {condition.get('answer')!r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", help="the saved JSON response")
    parser.add_argument("--condition", required=True, help="the one condition that was requested, e.g. ft_rag:v2")
    args = parser.parse_args(argv)
    try:
        with open(args.path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"FAIL could not read the response: {exc}")
        return 1
    checks = check_response(payload, condition=args.condition)
    print(report(payload, checks))
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
