#!/usr/bin/env python3
"""Install an already-trained LoRA adapter into the VM's Ollama, one version at a time.

    trained PEFT adapter -> validate -> convert to GGUF -> build a versioned
    Ollama model -> verify -> print the FINETUNED_OLLAMA_MODELS entry to add

    backend/.venv/bin/python scripts/install_finetuned_adapter.py \\
        --course css-350-spring-2026-n3h9 --version v1 \\
        --adapter ~/model_artifacts/css350-v1/adapter [--smoke] [--dry-run]

    # a GGUF converted elsewhere (e.g. on Tillicum) skips the conversion:
    backend/.venv/bin/python scripts/install_finetuned_adapter.py \\
        --course css-350-spring-2026-n3h9 --version v1 --gguf ~/model_artifacts/css350-v1/css350-v1-lora.gguf

Standard library only. What it refuses, deliberately: a course id or version
that fails the service's own rules; an adapter directory without
`adapter_config.json` and a weight file; an adapter whose
`base_model_name_or_path` is not the base this deployment serves; an existing
GGUF or Ollama tag for the same course and version, unless `--replace` is
passed; and any path that is not a real, existing directory or file.

What it never does: register or publish anything in PostgreSQL, change the
running mapping, restart a service, promote a model, or train. The registry
decides which version a course serves; this tool makes one version exist in
Ollama and prints the mapping entry an operator then adds with
`./scripts/aiswe_finetuned.sh set-mapping`.

`--dry-run` validates the adapter and prints every command it would run, and
needs neither Ollama nor llama.cpp on the host.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_HELPERS_PATH = REPO_ROOT / "scripts" / "lib" / "finetuned_local_helpers.py"

DEFAULT_BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_OLLAMA_BASE = "llama3.2:3b"
#: Which Ollama base a PEFT adapter trained on a given Hugging Face base may be
#: attached to. An adapter is a delta against specific weights; attaching it to
#: another model produces fluent nonsense with nothing on its face to show it.
COMPATIBLE_OLLAMA_BASES = {
    "meta-llama/Llama-3.2-3B-Instruct": ("llama3.2:3b", "llama3.2:3b-instruct-q4_K_M"),
}
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_ARTIFACTS_ROOT = "~/model_artifacts"
DEFAULT_LLAMA_CPP_DIRS = ("~/llama.cpp",)
DEFAULT_CONVERTER_VENVS = ("~/cpu-training-venv",)
CONVERTER_NAME = "convert_lora_to_gguf.py"
ADAPTER_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt")
SMOKE_QUESTION = "When does the course meet?"


def _load_local_helpers():
    spec = importlib.util.spec_from_file_location("finetuned_local_helpers", LOCAL_HELPERS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {LOCAL_HELPERS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = _load_local_helpers()


class InstallError(Exception):
    """A refusal. The message says what to change."""


# --------------------------------------------------------------------------- #
# External effects, in one place so tests can replace them
# --------------------------------------------------------------------------- #


class Runner:
    """Subprocesses and Ollama's HTTP API. Everything with a side effect."""

    def run(self, command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)

    def get_json(self, url: str, timeout: float = 10.0) -> Any:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - loopback Ollama
            return json.load(response)

    def post_json(self, url: str, body: dict[str, Any], timeout: float = 180.0) -> Any:
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.load(response)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _existing_dir(raw: str, what: str) -> Path:
    candidate = Path(os.path.expanduser(raw))
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise InstallError(f"{what} does not exist: {raw}") from exc
    if not resolved.is_dir():
        raise InstallError(f"{what} is not a directory: {resolved}")
    return resolved


def _existing_file(raw: str, what: str) -> Path:
    candidate = Path(os.path.expanduser(raw))
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise InstallError(f"{what} does not exist: {raw}") from exc
    if not resolved.is_file():
        raise InstallError(f"{what} is not a file: {resolved}")
    return resolved


def inspect_adapter(adapter_dir: Path) -> dict[str, Any]:
    """Read `adapter_config.json` and find the weight file; refuse anything partial."""
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise InstallError(f"Not a PEFT adapter directory (no adapter_config.json): {adapter_dir}")
    weight = next((adapter_dir / name for name in ADAPTER_WEIGHT_NAMES if (adapter_dir / name).is_file()), None)
    if weight is None:
        raise InstallError(
            f"Adapter directory has no weight file (expected one of {', '.join(ADAPTER_WEIGHT_NAMES)}): {adapter_dir}"
        )
    if weight.stat().st_size == 0:
        raise InstallError(f"Adapter weight file is empty: {weight}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise InstallError(f"adapter_config.json is not valid JSON: {config_path}") from exc
    if not isinstance(config, dict):
        raise InstallError(f"adapter_config.json is not a JSON object: {config_path}")
    peft_type = str(config.get("peft_type") or "").upper()
    if peft_type != "LORA":
        raise InstallError(f"Adapter is peft_type={config.get('peft_type')!r}, not LORA; Ollama loads LoRA adapters only.")
    base = config.get("base_model_name_or_path")
    if not isinstance(base, str) or not base.strip():
        raise InstallError("adapter_config.json names no base_model_name_or_path.")
    return {
        "dir": adapter_dir,
        "configPath": config_path,
        "weightPath": weight,
        "baseModel": base.strip(),
        "r": config.get("r"),
        "loraAlpha": config.get("lora_alpha"),
        "targetModules": sorted(config.get("target_modules") or []) if isinstance(config.get("target_modules"), list) else config.get("target_modules"),
        "peftType": peft_type,
    }


def check_base_compatibility(adapter_base: str, expected_base_id: str, ollama_base: str, *, allow_mismatch: bool) -> list[str]:
    """The adapter's base must be the base this deployment serves.

    Returns warnings (only when `allow_mismatch` waived a refusal).
    """
    warnings: list[str] = []
    if adapter_base != expected_base_id:
        message = (
            f"Adapter was trained on {adapter_base!r}, but this install expects {expected_base_id!r}. "
            "Pass --base-model-id to state the adapter's base explicitly, or --allow-base-mismatch to override."
        )
        if not allow_mismatch:
            raise InstallError(message)
        warnings.append("WARNING: " + message)
    compatible = COMPATIBLE_OLLAMA_BASES.get(adapter_base)
    if compatible is not None and helpers.normalize_ollama_model_name(ollama_base) not in {
        helpers.normalize_ollama_model_name(name) for name in compatible
    }:
        message = (
            f"Ollama base {ollama_base!r} is not a known match for {adapter_base!r} "
            f"(expected one of {', '.join(compatible)}). Pass --allow-base-mismatch to override."
        )
        if not allow_mismatch:
            raise InstallError(message)
        warnings.append("WARNING: " + message)
    return warnings


def find_converter(explicit: str | None, llama_cpp_dir: str | None) -> Path | None:
    """`convert_lora_to_gguf.py`, from the explicit path, a llama.cpp checkout, or the defaults."""
    if explicit:
        return _existing_file(explicit, "Converter script")
    candidates: list[Path] = []
    if llama_cpp_dir:
        candidates.append(Path(os.path.expanduser(llama_cpp_dir)) / CONVERTER_NAME)
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        candidates.append(Path(os.path.expanduser(env_dir)) / CONVERTER_NAME)
    candidates.extend(Path(os.path.expanduser(d)) / CONVERTER_NAME for d in DEFAULT_LLAMA_CPP_DIRS)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def find_converter_python(explicit: str | None) -> str:
    """The interpreter that has llama.cpp's Python requirements (gguf, torch, safetensors)."""
    if explicit:
        return str(_existing_file(explicit, "Converter interpreter"))
    env_venv = os.environ.get("CPU_TRAINING_VENV")
    candidates = ([Path(os.path.expanduser(env_venv)) / "bin" / "python"] if env_venv else []) + [
        Path(os.path.expanduser(d)) / "bin" / "python" for d in DEFAULT_CONVERTER_VENVS
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return sys.executable


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ollama_tags(runner: Runner, ollama_url: str) -> dict[str, str]:
    """`{name: digest}` for every model Ollama has, names in their tagged form."""
    try:
        payload = runner.get_json(f"{ollama_url}/api/tags")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise InstallError(f"Ollama at {ollama_url} did not answer /api/tags: {exc}") from exc
    tags: dict[str, str] = {}
    for item in (payload or {}).get("models", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
        if isinstance(name, str) and name.strip():
            try:
                tags[helpers.normalize_ollama_model_name(name)] = str(item.get("digest") or "")
            except helpers.CourseAdapterError:
                continue
    return tags


def render_modelfile(ollama_base: str, gguf_path: Path) -> str:
    # No PARAMETER lines: the service sends every decoding option per request.
    return f"FROM {ollama_base}\nADAPTER {gguf_path}\n"


def write_modelfile_atomically(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".Modelfile.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Everything the install will do, decided before anything is done."""
    course_id = helpers.validate_course_id(args.course)
    version = helpers.validate_model_version(args.version)
    tag = helpers.normalize_ollama_model_name(args.tag or helpers.default_ollama_tag(course_id, version))
    artifacts_root = Path(os.path.expanduser(args.artifacts_root)).resolve()
    out_dir = artifacts_root / course_id / version

    plan: dict[str, Any] = {
        "courseId": course_id,
        "version": version,
        "tag": tag,
        "ollamaBase": args.ollama_base,
        "ollamaUrl": args.ollama_url.rstrip("/"),
        "outDir": out_dir,
        "modelfile": out_dir / "Modelfile",
        "record": out_dir / "install-record.json",
        "replace": bool(args.replace),
        "smoke": bool(args.smoke),
        "warnings": [],
        "adapter": None,
        "convert": None,
        "gguf": None,
    }

    if args.gguf:
        plan["gguf"] = _existing_file(args.gguf, "GGUF file")
        if plan["gguf"].stat().st_size == 0:
            raise InstallError(f"GGUF file is empty: {plan['gguf']}")
        if args.adapter:
            adapter = inspect_adapter(_existing_dir(args.adapter, "Adapter directory"))
            plan["warnings"].extend(
                check_base_compatibility(adapter["baseModel"], args.base_model_id, args.ollama_base, allow_mismatch=args.allow_base_mismatch)
            )
            plan["adapter"] = adapter
        return plan

    if not args.adapter:
        raise InstallError("Pass --adapter <dir> (a trained PEFT adapter) or --gguf <file> (already converted).")
    adapter = inspect_adapter(_existing_dir(args.adapter, "Adapter directory"))
    plan["warnings"].extend(
        check_base_compatibility(adapter["baseModel"], args.base_model_id, args.ollama_base, allow_mismatch=args.allow_base_mismatch)
    )
    plan["adapter"] = adapter
    gguf = out_dir / helpers.gguf_filename(course_id, version)
    plan["gguf"] = gguf
    if gguf.exists() and not args.replace:
        raise InstallError(f"GGUF already exists: {gguf}. Pass --replace to overwrite it, or --gguf to reuse it.")

    converter = find_converter(args.converter, args.llama_cpp_dir)
    converter_python = find_converter_python(args.converter_python)
    command = [converter_python, str(converter) if converter else CONVERTER_NAME]
    if args.base_model_path:
        command += ["--base", str(_existing_dir(args.base_model_path, "Base model directory"))]
    else:
        command += ["--base-model-id", args.base_model_id]
    command += ["--outfile", str(gguf), "--outtype", args.outtype, str(adapter["dir"])]
    plan["convert"] = {"converter": converter, "python": converter_python, "outtype": args.outtype, "command": command}
    return plan


def describe_plan(plan: dict[str, Any]) -> str:
    lines = [
        f"course:      {plan['courseId']}",
        f"version:     {plan['version']}",
        f"ollama tag:  {plan['tag']}  (FROM {plan['ollamaBase']})",
        f"artifacts:   {plan['outDir']}",
    ]
    adapter = plan.get("adapter")
    if adapter:
        lines.append(
            f"adapter:     {adapter['dir']}  (base {adapter['baseModel']}, r={adapter['r']}, alpha={adapter['loraAlpha']}, "
            f"weights {adapter['weightPath'].name} {adapter['weightPath'].stat().st_size} bytes)"
        )
    convert = plan.get("convert")
    if convert:
        found = "found" if convert["converter"] else "NOT FOUND (pass --converter or --llama-cpp-dir)"
        lines.append(f"converter:   {convert['converter'] or CONVERTER_NAME} [{found}], python {convert['python']}, outtype {convert['outtype']}")
        lines.append(f"  $ {shlex.join(convert['command'])}")
    lines.append(f"gguf:        {plan['gguf']}")
    lines.append(f"  $ ollama create {plan['tag']} -f {plan['modelfile']}")
    if plan["smoke"]:
        lines.append(f"  then one short /api/chat against {plan['tag']}")
    lines.extend(plan["warnings"])
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def execute(plan: dict[str, Any], runner: Runner, *, dry_run: bool, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Run the plan. Returns the install record (also written next to the GGUF)."""
    log(describe_plan(plan))
    if dry_run:
        log("dry run: nothing converted, created or written.")
        log(next_steps(plan))
        return {"dryRun": True, "mappingEntry": mapping_entry(plan)}

    convert = plan.get("convert")
    if convert and convert["converter"] is None:
        raise InstallError(
            f"{CONVERTER_NAME} not found. Pass --converter <path> or --llama-cpp-dir <checkout>, "
            "or convert on a host that has llama.cpp and pass the result with --gguf."
        )

    tags = ollama_tags(runner, plan["ollamaUrl"])
    if plan["tag"] in tags and not plan["replace"]:
        raise InstallError(
            f"Ollama already has {plan['tag']} (digest {tags[plan['tag']][:12]}). A version is built once; "
            "pass --replace only if you mean to rebuild this exact version."
        )
    base_tag = helpers.normalize_ollama_model_name(plan["ollamaBase"])
    if base_tag not in tags:
        raise InstallError(f"Ollama does not have the base model {base_tag}; pull it first (ollama pull {plan['ollamaBase']}).")

    plan["outDir"].mkdir(parents=True, exist_ok=True)

    if convert:
        log(f"converting: {shlex.join(convert['command'])}")
        result = runner.run(convert["command"])
        if result.returncode != 0:
            raise InstallError(
                f"Conversion failed (exit {result.returncode}).\n{(result.stderr or result.stdout or '').strip()[-2000:]}"
            )
        if not plan["gguf"].is_file() or plan["gguf"].stat().st_size == 0:
            raise InstallError(f"Converter exited 0 but wrote no GGUF at {plan['gguf']}")

    modelfile_text = render_modelfile(plan["ollamaBase"], plan["gguf"])
    write_modelfile_atomically(plan["modelfile"], modelfile_text)
    # `ollama create` takes the bare name; `/api/tags` reports it as name:latest.
    create_name = plan["tag"][: -len(":latest")] if plan["tag"].endswith(":latest") else plan["tag"]
    log(f"creating: ollama create {create_name} -f {plan['modelfile']}")
    result = runner.run(["ollama", "create", create_name, "-f", str(plan["modelfile"])])
    if result.returncode != 0:
        raise InstallError(f"ollama create failed (exit {result.returncode}).\n{(result.stderr or result.stdout or '').strip()[-2000:]}")

    tags_after = ollama_tags(runner, plan["ollamaUrl"])
    if plan["tag"] not in tags_after:
        raise InstallError(f"ollama create exited 0 but {plan['tag']} is not in /api/tags.")
    digest = tags_after[plan["tag"]]
    log(f"verified: {plan['tag']} exists (digest {digest[:12] or 'unknown'})")

    smoke: dict[str, Any] | None = None
    if plan["smoke"]:
        smoke = smoke_generate(runner, plan["ollamaUrl"], plan["tag"])
        log(f"smoke: {smoke['characters']} characters in {smoke['seconds']:.1f}s")

    record = build_record(plan, digest=digest, smoke=smoke)
    plan["record"].write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"record: {plan['record']}")
    log(next_steps(plan))
    return record


def smoke_generate(runner: Runner, ollama_url: str, tag: str) -> dict[str, Any]:
    """One short greedy generation. Proves the tag loads; says nothing about quality."""
    import time

    body = {
        "model": tag,
        "messages": [{"role": "user", "content": SMOKE_QUESTION}],
        "stream": False,
        "options": {"num_predict": 32, "temperature": 0, "seed": 360},
    }
    started = time.perf_counter()
    try:
        payload = runner.post_json(f"{ollama_url}/api/chat", body)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise InstallError(f"Smoke generation against {tag} failed: {exc}") from exc
    elapsed = time.perf_counter() - started
    content = ((payload or {}).get("message") or {}).get("content") if isinstance(payload, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise InstallError(f"Smoke generation against {tag} returned no text.")
    return {"characters": len(content.strip()), "seconds": elapsed}


def build_record(plan: dict[str, Any], *, digest: str, smoke: dict[str, Any] | None) -> dict[str, Any]:
    adapter = plan.get("adapter")
    convert = plan.get("convert")
    gguf: Path = plan["gguf"]
    return {
        "courseId": plan["courseId"],
        "version": plan["version"],
        "ollamaTag": plan["tag"],
        "ollamaBase": plan["ollamaBase"],
        "ollamaDigest": digest or None,
        "installedAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "gguf": {"path": str(gguf), "bytes": gguf.stat().st_size, "sha256": sha256_of(gguf)},
        "adapter": (
            {
                "path": str(adapter["dir"]),
                "baseModel": adapter["baseModel"],
                "r": adapter["r"],
                "loraAlpha": adapter["loraAlpha"],
                "targetModules": adapter["targetModules"],
                "weightFile": adapter["weightPath"].name,
                "weightBytes": adapter["weightPath"].stat().st_size,
                "weightSha256": sha256_of(adapter["weightPath"]),
                "configSha256": sha256_of(adapter["configPath"]),
            }
            if adapter
            else None
        ),
        "conversion": (
            {"converter": str(convert["converter"]), "python": convert["python"], "outtype": convert["outtype"], "command": convert["command"]}
            if convert
            else None
        ),
        "smoke": smoke,
        "mappingEntry": mapping_entry(plan),
    }


def mapping_entry(plan: dict[str, Any]) -> str:
    return helpers.format_mapping_entry(plan["courseId"], plan["version"], plan["tag"])


def next_steps(plan: dict[str, Any]) -> str:
    return (
        "\nNothing in PostgreSQL was changed and nothing is served yet. To serve this version:\n"
        f"  ./scripts/aiswe_finetuned.sh set-mapping {plan['courseId']} {plan['version']} {plan['tag']}\n"
        "  ./scripts/aiswe_finetuned.sh restart\n"
        f"FINETUNED_OLLAMA_MODELS entry: {mapping_entry(plan)}"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--course", required=True, help="course id, e.g. css-350-spring-2026-n3h9")
    parser.add_argument("--version", required=True, help="registered model version, e.g. v1")
    source = parser.add_argument_group("source (one of)")
    source.add_argument("--adapter", help="trained PEFT adapter directory (adapter_config.json + weights)")
    source.add_argument("--gguf", help="an already-converted LoRA GGUF; skips conversion")
    parser.add_argument("--tag", help=f"Ollama model name (default: <course>-ft-<version>, e.g. css350-ft-v1)")
    parser.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS_ROOT, help=f"where GGUF, Modelfile and record go (default {DEFAULT_ARTIFACTS_ROOT}/<course>/<version>/)")
    conv = parser.add_argument_group("conversion")
    conv.add_argument("--converter", help=f"path to llama.cpp's {CONVERTER_NAME}")
    conv.add_argument("--llama-cpp-dir", help=f"llama.cpp checkout containing {CONVERTER_NAME} (also $LLAMA_CPP_DIR; default ~/llama.cpp)")
    conv.add_argument("--converter-python", help="interpreter with gguf/torch/safetensors (default: $CPU_TRAINING_VENV or ~/cpu-training-venv, else this one)")
    conv.add_argument("--base-model-path", help="local Hugging Face directory of the base model for the converter (--base)")
    conv.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID, help=f"Hugging Face id the adapter was trained on (default {DEFAULT_BASE_MODEL_ID})")
    conv.add_argument("--outtype", default="f16", choices=("f16", "f32", "bf16", "q8_0", "auto"), help="GGUF tensor type (default f16)")
    ollama = parser.add_argument_group("ollama")
    ollama.add_argument("--ollama-base", default=DEFAULT_OLLAMA_BASE, help=f"Modelfile FROM line (default {DEFAULT_OLLAMA_BASE})")
    ollama.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--allow-base-mismatch", action="store_true", help="proceed although the adapter's base does not match")
    parser.add_argument("--replace", action="store_true", help="overwrite an existing GGUF / Ollama tag for this course and version")
    parser.add_argument("--smoke", action="store_true", help="after creating the model, run one short generation")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan; run nothing")
    return parser


def main(argv: list[str] | None = None, runner: Runner | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_plan(args)
        execute(plan, runner or Runner(), dry_run=args.dry_run)
    except (InstallError, helpers.CourseAdapterError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
