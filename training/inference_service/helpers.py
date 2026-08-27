"""Pure helpers for the fine-tuned inference service (no GPU imports).

Per-course serving
------------------
The service used to load exactly one adapter, from one path, with no course
identity anywhere in it: `promote_qlora_adapter.sh` copied the newest adapter
over `training_outputs/css-360-qlora/adapter` and the service served whatever
was there. That was survivable while one course existed. With CSS 350 and CSS
360 both trained it is not survivable at all — promoting one course's adapter
silently replaces the other's, and a request carries nothing that could detect
it, because a request carries no course.

So adapters are published per course and per version into a serving root:

    <serving root>/<courseId>/<version>/adapter/     the PEFT adapter
    <serving root>/<courseId>/current.json           which version is current

and a request names the course it is for. The pointer is a small JSON file
rather than a symlink because it is written atomically with `os.replace` on the
same filesystem, it can carry the run id and the source reference alongside the
version, and it reads identically from a login node, a compute node and a test.

Course ids are validated with the same rule the backend and the frontend use.
A path is only ever built from a validated id and a validated version, so a
request cannot address anything outside the serving root.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_SEED = 360

#: How many course adapters may be resident at once.
#:
#: LoRA adapters for this configuration are ~47 MB against a ~2.5 GB 4-bit base
#: model, so several fit comfortably beside one base. The bound exists so that a
#: term with a dozen courses cannot walk the GPU into an out-of-memory error one
#: request at a time; four is enough to cover a demo that switches back and forth
#: without reloading.
DEFAULT_MAX_LOADED_ADAPTERS = 4

ADAPTER_DIRNAME = "adapter"
CURRENT_POINTER_FILENAME = "current.json"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"
ADAPTER_WEIGHT_NAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_model.pt",
)

COURSE_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
VERSION_RE = re.compile(r"^v[0-9]+$")


class CourseAdapterError(Exception):
    """No usable adapter for the course and version asked for."""


def validate_course_id(course_id: str) -> str:
    """The same rule the backend enforces, restated where a path is built.

    Not defence in depth for its own sake: this value becomes a directory name
    under the serving root, and the service must not depend on the backend
    having checked it. `..` and `/` fail the pattern outright.
    """
    value = (course_id or "").strip()
    if not value or ".." in value or "/" in value or not COURSE_ID_RE.fullmatch(value):
        raise CourseAdapterError(
            "Invalid courseId: {0!r}. Expected lowercase letters, digits and "
            "hyphens, e.g. css-360-winter-2026-a7rp.".format(course_id)
        )
    return value


def validate_model_version(version: str) -> str:
    value = (version or "").strip()
    if not VERSION_RE.fullmatch(value):
        raise CourseAdapterError(
            "Invalid model version: {0!r}. Expected v1, v2, ….".format(version)
        )
    return value


def resolve_serving_root() -> Path:
    """Where published per-course adapters live."""
    raw = (os.environ.get("SERVING_ROOT") or "").strip()
    if raw:
        return Path(raw)
    user = os.environ.get("USER", "USER")
    return Path("/gpfs/projects/simswe/{0}/training_outputs/serving".format(user))


def course_serving_dir(course_id: str, *, root: Optional[Path] = None) -> Path:
    base = root if root is not None else resolve_serving_root()
    return base / validate_course_id(course_id)


def course_adapter_dir(
    course_id: str, version: str, *, root: Optional[Path] = None
) -> Path:
    """The adapter directory for exactly one course and one version."""
    return (
        course_serving_dir(course_id, root=root)
        / validate_model_version(version)
        / ADAPTER_DIRNAME
    )


def adapter_is_loadable(path: Path) -> bool:
    """Whether a directory holds a PEFT adapter this service can load.

    The format check that matters: training writes `adapter_config.json` plus
    `adapter_model.safetensors`, and `PeftModel.load_adapter` reads exactly that
    pair. No conversion step exists or is needed — the artifact training produces
    is the artifact serving consumes.
    """
    if not path.is_dir():
        return False
    if not (path / ADAPTER_CONFIG_FILENAME).is_file():
        return False
    return any((path / name).is_file() for name in ADAPTER_WEIGHT_NAMES)


def read_current_pointer(
    course_id: str, *, root: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """The `current.json` a course was published with, or None."""
    path = course_serving_dir(course_id, root=root) / CURRENT_POINTER_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    version = payload.get("version")
    if not isinstance(version, str) or not VERSION_RE.fullmatch(version.strip()):
        return None
    payload["version"] = version.strip()
    return payload


def list_course_versions(course_id: str, *, root: Optional[Path] = None) -> List[str]:
    """Published versions for one course, newest-numbered last."""
    directory = course_serving_dir(course_id, root=root)
    if not directory.is_dir():
        return []
    versions = [
        entry.name
        for entry in directory.iterdir()
        if entry.is_dir()
        and VERSION_RE.fullmatch(entry.name)
        and adapter_is_loadable(entry / ADAPTER_DIRNAME)
    ]
    return sorted(versions, key=lambda name: int(name[1:]))


def resolve_course_adapter(
    course_id: str,
    version: Optional[str] = None,
    *,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Which adapter answers for this course, and where it is.

    An explicit version wins, because the backend resolves the course's current
    version from PostgreSQL — the system of record — and sending it makes the
    two sides verifiable against each other. `current.json` is the fallback for
    a caller that did not resolve one, and the highest published version is the
    last resort for a course published before pointers existed.

    Raises rather than falling back to another course's adapter or to a
    course-agnostic default. There is no answer to "serve CSS 350" that involves
    loading something else.
    """
    safe_course_id = validate_course_id(course_id)
    source = "requested"

    if version:
        safe_version = validate_model_version(version)
    else:
        pointer = read_current_pointer(safe_course_id, root=root)
        if pointer is not None:
            safe_version = pointer["version"]
            source = "current.json"
        else:
            published = list_course_versions(safe_course_id, root=root)
            if not published:
                raise CourseAdapterError(
                    'No fine-tuned adapter is published for course "{0}". '
                    "Publish one with ./training/promote_qlora_adapter.sh "
                    "--course {0} --version <vN> <adapter-path>.".format(
                        safe_course_id
                    )
                )
            safe_version = published[-1]
            source = "highest published"

    path = course_adapter_dir(safe_course_id, safe_version, root=root)
    if not adapter_is_loadable(path):
        raise CourseAdapterError(
            'Course "{0}" has no loadable adapter for version {1}. Published '
            "versions: {2}.".format(
                safe_course_id,
                safe_version,
                ", ".join(list_course_versions(safe_course_id, root=root)) or "none",
            )
        )

    return {
        "courseId": safe_course_id,
        "version": safe_version,
        "path": path,
        "adapterKey": adapter_key(safe_course_id, safe_version),
        "versionSource": source,
    }


def adapter_key(course_id: str, version: str) -> str:
    """The name one adapter is registered under inside the loaded model.

    Course and version together. PEFT keeps adapters in a flat namespace on the
    base model, so a key that named only the course would make a promotion
    indistinguishable from the version it replaced, and a demo would keep
    serving the old weights until the process restarted.
    """
    return "{0}@{1}".format(course_id, version)


def list_available_courses(*, root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Every course with at least one published adapter."""
    base = root if root is not None else resolve_serving_root()
    if not base.is_dir():
        return []

    courses: List[Dict[str, Any]] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        try:
            course_id = validate_course_id(entry.name)
        except CourseAdapterError:
            continue
        versions = list_course_versions(course_id, root=base)
        if not versions:
            continue
        pointer = read_current_pointer(course_id, root=base)
        courses.append(
            {
                "courseId": course_id,
                "versions": versions,
                "currentVersion": (pointer or {}).get("version") or versions[-1],
            }
        )
    return courses


def resolve_max_loaded_adapters() -> int:
    raw = (os.environ.get("MAX_LOADED_ADAPTERS") or "").strip()
    if not raw:
        return DEFAULT_MAX_LOADED_ADAPTERS
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            "Invalid MAX_LOADED_ADAPTERS value: {0!r}".format(raw)
        ) from exc
    if value < 1:
        raise RuntimeError("MAX_LOADED_ADAPTERS must be at least 1.")
    return value


def resolve_model_id() -> str:
    return (os.environ.get("MODEL_ID") or DEFAULT_MODEL_ID).strip()


def resolve_port() -> int:
    raw = (os.environ.get("INFERENCE_PORT") or os.environ.get("PORT") or "8001").strip()
    try:
        port = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid INFERENCE_PORT/PORT value: {raw!r}") from exc
    if port < 1 or port > 65535:
        raise RuntimeError(f"Port out of range: {port}")
    return port


def resolve_session_deadline() -> Optional[float]:
    """Unix time this service should stop serving at, if one was set.

    `SERVICE_DEADLINE_EPOCH` is written by the start script from the Slurm wall
    clock. The allocation ends then regardless; having the process know the same
    moment lets `/health` report a session as expiring rather than a caller
    discovering it as a connection reset.
    """
    raw = (os.environ.get("SERVICE_DEADLINE_EPOCH") or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def assert_hf_auth_available() -> None:
    token_path = (os.environ.get("HF_TOKEN_PATH") or "").strip()
    if token_path and Path(token_path).is_file() and Path(token_path).stat().st_size > 0:
        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
            token = Path(token_path).read_text(encoding="utf-8").strip()
            if token:
                os.environ["HF_TOKEN"] = token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        return
    if (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip():
        return
    try:
        from huggingface_hub import HfFolder

        cached = HfFolder.get_token()
        if cached:
            return
    except Exception:  # noqa: BLE001
        pass
    raise RuntimeError(
        "Hugging Face authentication is unavailable. "
        "Set HF_TOKEN_PATH to a non-empty token file, or export HF_TOKEN."
    )


def validate_question(question: str) -> str:
    cleaned = (question or "").strip()
    if not cleaned:
        raise ValueError("question must be a non-blank string")
    return cleaned
