#!/usr/bin/env python3
"""Verify VM-local fine-tuned inference on the UWB VM, end to end, in one command.

    backend/.venv/bin/python scripts/verify_finetuned_production.py
    AISWE_VERIFY_PASSWORD=... backend/.venv/bin/python scripts/verify_finetuned_production.py --admin-email you@uw.edu

Prints one PASS / FAIL / SKIP line per check and exits 1 if anything failed.
Unauthenticated checks always run: the unit, the port owner, Ollama, the
mapping, the service's `/health` and `/generate`, and the backend's health.
The checks that go through the backend's classroom routes (Fine-Tuned,
Fine-Tuned + RAG, Base, RAG, and the admin-only fine-tuned health) need an
administrator session and are SKIP, never PASS, without one. The password is
read from `AISWE_VERIFY_PASSWORD` or an interactive prompt, held in memory
for this run, and the session is logged out at the end. No answer text is
printed; only lengths, versions and timings.

Nothing here writes to production: no mapping change, no restart, no
registry change. Standard library only.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LIB = REPO_ROOT / "scripts" / "lib"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LIB / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: dataclasses under `from __future__ import
    # annotations` resolve their types through sys.modules[cls.__module__].
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


checks = _load("finetuned_production_checks")
local = _load("finetuned_local_helpers")

UNIT = "aiswe-finetuned"


class HostFacts:
    """Reads of the host that are not HTTP: listeners, unit state, the env file."""

    def listeners_text(self) -> str | None:
        if shutil.which("ss"):
            result = subprocess.run(["ss", "-H", "-ltnp"], capture_output=True, text=True, check=False)
            return result.stdout
        return None

    def unit_property(self, unit: str, verb: str) -> str | None:
        if not shutil.which("systemctl"):
            return None
        result = subprocess.run(["systemctl", "--user", verb, unit], capture_output=True, text=True, check=False)
        return (result.stdout or "").strip() or (result.stderr or "").strip()

    def env_text(self, path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None


def run(args: argparse.Namespace, *, facts: HostFacts, transport: Any = None, credentials: tuple[str, str] | None) -> list[checks.Check]:
    results: list[checks.Check] = []

    def add(status: str, name: str, detail: str = "") -> None:
        results.append((status, name, detail))

    # ---- host: unit, port, Ollama, mapping ---------------------------------
    for verb, want in (("is-enabled", "enabled"), ("is-active", "active")):
        value = facts.unit_property(UNIT, verb)
        if value is None:
            add("SKIP", f"{UNIT} {verb}", "no systemctl on this host")
        else:
            add("PASS" if value == want else "FAIL", f"{UNIT} {verb}", value)

    listeners = facts.listeners_text()
    port = int(args.service_url.rsplit(":", 1)[1].rstrip("/"))
    if listeners is None:
        add("SKIP", f"port {port} owner", "no `ss` on this host")
    else:
        owner = local.classify_port_owner(local.parse_ss_listeners(listeners), port)
        if owner["kind"] == "service":
            add("PASS", f"port {port} owned by the local service", f"{owner['process']} pid {owner['pid']}")
        elif owner["kind"] == "tunnel":
            add("FAIL", f"port {port} owned by an SSH forward (Tillicum fallback), not the local service", f"pid {owner['pid']}")
        elif owner["kind"] == "free":
            add("FAIL", f"port {port} has no listener", "")
        else:
            add("FAIL", f"port {port} owned by something else", f"{owner['process']} pid {owner['pid']}")

    session_free = checks.BackendSession(args.service_url, transport, timeout=args.timeout)
    ollama = checks.BackendSession(args.ollama_url, transport, timeout=15)
    tags_result = ollama.get("/api/tags")
    present: set[str] = set()
    if tags_result.ok and isinstance(tags_result.body, dict):
        for item in tags_result.body.get("models") or []:
            name = item.get("name") or item.get("model") if isinstance(item, dict) else None
            if isinstance(name, str):
                try:
                    present.add(local.normalize_ollama_model_name(name))
                except local.CourseAdapterError:
                    pass
        add("PASS", "ollama reachable", f"{len(present)} models")
    else:
        add("FAIL", "ollama reachable", tags_result.error or f"HTTP {tags_result.status}")

    env_path = Path(os.path.expanduser(args.env_file))
    env_text = facts.env_text(env_path)
    mapped: list[tuple[str, str, str]] = []
    if env_text is None:
        add("FAIL", "mapping file readable", str(env_path))
    else:
        summary = local.validate_env_text(env_text)
        if summary["error"]:
            add("FAIL", "mapping valid", summary["error"])
        else:
            mapped = [(e["courseId"], e["version"], e["ollamaModel"]) for e in summary["entries"]]
            add("PASS" if mapped else "FAIL", "mapping valid", f"{len(mapped)} entries" if mapped else "empty mapping")
        for course, version, model in mapped:
            if not tags_result.ok:
                add("SKIP", f"ollama has {model}", "ollama unreachable")
            else:
                add("PASS" if model in present else "FAIL", f"ollama has {model}", f"{course} {version}")

    expected_courses = list(args.course or checks.KNOWN_COURSES)
    versions_by_course: dict[str, list[str]] = {}
    for course, version, _model in mapped:
        versions_by_course.setdefault(course, []).append(version)
    for course in expected_courses:
        add("PASS" if course in versions_by_course else "FAIL", f"{course} is mapped", ", ".join(versions_by_course.get(course, [])) or "no entry")

    # ---- the service itself ------------------------------------------------
    health = session_free.get("/health", timeout=15)
    body = health.body if isinstance(health.body, dict) else {}
    if not health.ok:
        add("FAIL", "service /health", health.error or f"HTTP {health.status}")
    else:
        add("PASS" if body.get("status") == "ok" else "FAIL", "service /health status=ok", str(body.get("status")) + (f": {body.get('detail')}" if body.get("detail") else ""))
        add("PASS" if body.get("adapterLoaded") is True else "FAIL", "service /health adapterLoaded=true", str(body.get("adapterLoaded")))
        reported = {(c.get("courseId"), v) for c in body.get("courses") or [] if isinstance(c, dict) for v in (c.get("versions") or [])}
        for course, version, _model in mapped:
            add("PASS" if (course, version) in reported else "FAIL", f"service reports {course} {version} servable", "")

    if args.skip_generate:
        add("SKIP", "direct /generate", "--skip-generate")
    else:
        for course, version, _model in mapped:
            if course not in expected_courses:
                continue
            result = session_free.post("/generate", {"courseId": course, "modelVersion": version, "question": checks.PROBE_QUESTION})
            ok, detail = checks.judge_direct_generation(result, course_id=course, version=version)
            add("PASS" if ok else "FAIL", f"direct /generate {course} {version}", detail)

    # ---- the backend -------------------------------------------------------
    backend = checks.BackendSession(args.backend_url, transport, timeout=args.timeout)
    api_health = backend.get("/api/health", timeout=15)
    add("PASS" if api_health.ok and isinstance(api_health.body, dict) and api_health.body.get("status") == "ok" else "FAIL", "backend /api/health", api_health.error or f"HTTP {api_health.status}")

    if credentials is None:
        add("SKIP", "backend /api/fine-tuned/health", "no administrator credentials (set --admin-email and AISWE_VERIFY_PASSWORD)")
        for course in expected_courses:
            for mode in ("fineTuned", "fineTunedRag", "base", "rag"):
                add("SKIP", f"backend {mode} {course}", "no administrator credentials")
        return results

    login = backend.login(*credentials)
    if not login.ok or not backend.signed_in:
        add("FAIL", "administrator sign-in", login.error or f"HTTP {login.status}")
        for course in expected_courses:
            for mode in ("fineTuned", "fineTunedRag", "base", "rag"):
                add("SKIP", f"backend {mode} {course}", "sign-in failed")
        return results
    try:
        ft_health = backend.get("/api/fine-tuned/health", timeout=15)
        hb = ft_health.body if isinstance(ft_health.body, dict) else {}
        add(
            "PASS" if ft_health.ok and hb.get("status") == "ok" and hb.get("adapterLoaded") is True else "FAIL",
            "backend /api/fine-tuned/health",
            f"status={hb.get('status')} adapterLoaded={hb.get('adapterLoaded')} courses={[c.get('courseId') for c in hb.get('courses') or [] if isinstance(c, dict)]}" if ft_health.ok else (ft_health.error or f"HTTP {ft_health.status}"),
        )
        if args.skip_generate:
            add("SKIP", "backend generation", "--skip-generate")
        else:
            for course in expected_courses:
                for mode in ("fineTuned", "fineTunedRag", "base", "rag"):
                    result = checks.generate(backend, mode, course)
                    ok, detail = checks.judge_generation(result, course_id=course, mode=mode, allowed_versions=versions_by_course.get(course) if mode.startswith("fineTuned") else None)
                    add("PASS" if ok else "FAIL", f"backend {mode} {course}", detail)
    finally:
        backend.logout()
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend-url", default=checks.DEFAULT_BACKEND_URL)
    parser.add_argument("--service-url", default=checks.DEFAULT_SERVICE_URL)
    parser.add_argument("--ollama-url", default=checks.DEFAULT_OLLAMA_URL)
    parser.add_argument("--env-file", default=checks.DEFAULT_ENV_FILE, help="the unit's environment file (the mapping)")
    parser.add_argument("--course", action="append", help="course id to verify (default: CSS 350 and CSS 360)")
    parser.add_argument("--admin-email", help=f"administrator email for the backend checks (or ${checks.EMAIL_ENV}); password from ${checks.PASSWORD_ENV} or a prompt")
    parser.add_argument("--timeout", type=float, default=180.0, help="per-generation timeout in seconds")
    parser.add_argument("--skip-generate", action="store_true", help="health checks only; no model is asked anything")
    return parser


def main(argv: list[str] | None = None, *, facts: HostFacts | None = None, transport: Any = None) -> int:
    args = build_parser().parse_args(argv)
    credentials = checks.resolve_admin_credentials(args.admin_email)
    results = run(args, facts=facts or HostFacts(), transport=transport, credentials=credentials)
    return 0 if checks.print_checks(results) else 1


if __name__ == "__main__":
    sys.exit(main())
