#!/usr/bin/env python3
"""Load the classroom models into Ollama before students arrive, and keep them there.

    backend/.venv/bin/python scripts/warm_classroom_models.py \\
        --base-model llama3.2:3b \\
        --finetuned-model css360e-v1:latest --keep-alive 4h

A model's first request after it was unloaded pays the load (seconds on the
CPU VM), and the first students of a class are the ones who pay it. This
sends each model an empty chat, which Ollama answers by loading the model and
generating nothing, with the exact runner options the classroom requests use
— above all the same `num_ctx`. A load with a different context size would
be undone by the first real request, which Ollama would answer by loading
the model a second time.

Then it checks, from `/api/ps`, that the base model, the embedding model and
the (last) course model are actually resident on the servers they will be
asked on, with the classroom context size, and exits 1 if not. Nothing is
generated, nothing is written, no student data is involved.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def classroom_num_ctx() -> int:
    """`GROUNDED_NUM_CTX`, read from the module every classroom request uses.

    `grounded_generation.py` is dependency-free by design, so it loads
    without the backend's virtualenv or configuration.
    """
    path = REPO_ROOT / "backend" / "app" / "grounded_generation.py"
    spec = importlib.util.spec_from_file_location("grounded_generation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["grounded_generation"] = module
    spec.loader.exec_module(module)
    return int(module.GROUNDED_NUM_CTX)


def post(url: str, body: dict[str, Any], *, timeout: float) -> tuple[int, Any, float]:
    """(status, parsed body, seconds). Status 0 means no answer at all (down, refused, timed out)."""
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - loopback
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read(), exc.code
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return 0, {"error": f"no answer ({reason})"}, time.perf_counter() - started
    elapsed = time.perf_counter() - started
    try:
        return status, json.loads(raw.decode("utf-8") or "null"), elapsed
    except ValueError:
        return status, None, elapsed


def get(url: str, *, timeout: float = 5.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - loopback
        return json.loads(response.read().decode("utf-8"))


def warm_chat_model(ollama_url: str, model: str, *, num_ctx: int, keep_alive: str, timeout: float) -> str:
    status, body, elapsed = post(
        f"{ollama_url.rstrip('/')}/api/chat",
        {"model": model, "messages": [], "keep_alive": keep_alive, "options": {"num_ctx": num_ctx}},
        timeout=timeout,
    )
    if status != 200:
        error = body.get("error") if isinstance(body, dict) else body
        return f"FAIL {model} @ {ollama_url}: HTTP {status} {error}"
    return f"ok   {model} @ {ollama_url}: loaded in {elapsed:.1f}s (num_ctx {num_ctx}, keep_alive {keep_alive})"


def warm_embed_model(ollama_url: str, model: str, *, keep_alive: str, timeout: float) -> str:
    status, body, elapsed = post(
        f"{ollama_url.rstrip('/')}/api/embed",
        {"model": model, "input": ["warm"], "keep_alive": keep_alive},
        timeout=timeout,
    )
    if status != 200:
        error = body.get("error") if isinstance(body, dict) else body
        return f"FAIL {model} @ {ollama_url}: HTTP {status} {error}"
    return f"ok   {model} @ {ollama_url}: loaded in {elapsed:.1f}s (keep_alive {keep_alive})"


def resident(ollama_url: str) -> list[dict[str, Any]] | None:
    """`/api/ps` models, or None when the server cannot be asked."""
    try:
        body = get(f"{ollama_url.rstrip('/')}/api/ps")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    models = body.get("models") if isinstance(body, dict) else None
    return [m for m in models if isinstance(m, dict)] if isinstance(models, list) else []


def _same_model(a: str, b: str) -> bool:
    def norm(name: str) -> str:
        return name if ":" in name else f"{name}:latest"

    return norm(a) == norm(b)


def check_resident(ollama_url: str, model: str, *, num_ctx: int | None) -> str:
    """PASS when `model` is loaded on `ollama_url` (with `num_ctx`, for chat models)."""
    models = resident(ollama_url)
    if models is None:
        return f"FAIL {model} @ {ollama_url}: cannot list resident models"
    for m in models:
        if _same_model(str(m.get("name") or ""), model):
            ctx = m.get("context_length")
            if num_ctx is not None and isinstance(ctx, int) and ctx != num_ctx:
                return f"FAIL {model} @ {ollama_url}: resident with context {ctx}, classroom requests use {num_ctx} (the first one would reload it)"
            return f"PASS {model} @ {ollama_url}: resident until {m.get('expires_at', '?')}"
    others = ", ".join(str(m.get("name")) for m in models) or "nothing"
    return f"FAIL {model} @ {ollama_url}: not resident (resident: {others})"


def describe_resident(ollama_url: str) -> list[str]:
    models = resident(ollama_url)
    if models is None:
        return [f"  {ollama_url}: cannot list resident models"]
    if not models:
        return [f"  {ollama_url}: nothing resident"]
    return [
        f"  {ollama_url}: {m.get('name')} until {m.get('expires_at', '?')} "
        f"({(m.get('size') or 0) / 2**30:.1f} GiB, ctx {m.get('context_length', '?')})"
        for m in models
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-ollama", default="http://127.0.0.1:11434", help="The Ollama the backend uses (default %(default)s).")
    parser.add_argument("--base-model", default="llama3.2:3b", help="OLLAMA_MODEL on the backend (default %(default)s).")
    parser.add_argument("--embed-model", default="nomic-embed-text", help="OLLAMA_EMBEDDING_MODEL on the backend (default %(default)s).")
    parser.add_argument("--finetuned-ollama", default="http://127.0.0.1:11435", help="The Ollama the fine-tuned service uses: its OLLAMA_BASE_URL (default %(default)s, the separate fine-tuned server).")
    parser.add_argument("--finetuned-model", action="append", default=[], help="An activated course model tag (repeatable).")
    parser.add_argument("--keep-alive", default="30m", help="How long Ollama keeps each model after its last request (default %(default)s).")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-load timeout in seconds (default %(default)s).")
    args = parser.parse_args(argv)

    num_ctx = classroom_num_ctx()
    lines: list[str] = []

    def report(line: str) -> None:
        lines.append(line)
        print(line, flush=True)

    report(warm_chat_model(args.base_ollama, args.base_model, num_ctx=num_ctx, keep_alive=args.keep_alive, timeout=args.timeout))
    report(warm_embed_model(args.base_ollama, args.embed_model, keep_alive=args.keep_alive, timeout=args.timeout))
    for model in args.finetuned_model:
        report(warm_chat_model(args.finetuned_ollama, model, num_ctx=num_ctx, keep_alive=args.keep_alive, timeout=args.timeout))

    if len(args.finetuned_model) > 1:
        print(
            "NOTE: course models built FROM the same base share one weights file, and one Ollama keeps\n"
            "      only one of them loaded at a time; only the last one listed can be resident now."
        )
    if args.finetuned_model and args.finetuned_ollama.rstrip("/") == args.base_ollama.rstrip("/"):
        print(
            "NOTE: the fine-tuned models share an Ollama with the base model, so loading one unloaded\n"
            "      the other. Serve them from a second Ollama (docs/classroom-capacity.md)."
        )

    # Verify what is actually loaded, rather than trusting each load call: a
    # later load can have unloaded an earlier one.
    print("Verify:")
    report(check_resident(args.base_ollama, args.base_model, num_ctx=num_ctx))
    report(check_resident(args.base_ollama, args.embed_model, num_ctx=None))
    if args.finetuned_model:
        report(check_resident(args.finetuned_ollama, args.finetuned_model[-1], num_ctx=num_ctx))

    print("Resident now:")
    for url in dict.fromkeys([args.base_ollama, args.finetuned_ollama]):
        for line in describe_resident(url):
            print(line)
    return 1 if any(line.startswith("FAIL") for line in lines) else 0


if __name__ == "__main__":
    sys.exit(main())
