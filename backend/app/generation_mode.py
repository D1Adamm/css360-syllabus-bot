"""Where the four conditions are generated: on the VM (normal) or on a Tillicum GPU.

Two modes
---------
- **vm** (normal, and the default when nothing says otherwise): Base and RAG on
  the VM's Ollama (`OLLAMA_BASE_URL`), Fine-Tuned and Fine-Tuned + RAG on the
  VM-local fine-tuned service (`FINETUNED_SERVICE_URL`).
- **gpu** (classroom GPU mode): all four on the Tillicum GPU service, reached
  through an SSH tunnel on a loopback port. Base and RAG ask it for the base
  model with every adapter off, the fine-tuned conditions for the course's
  adapter. Retrieval, the prompts, the queue, auth and the database stay here.

What does not change between modes: which prompt each condition gets, the
decoding settings (checked on every GPU answer, see `gpu_generation`), and the
course version the fine-tuned conditions answer from.

How a mode is chosen
--------------------
A small JSON file (`GENERATION_MODE_FILE`, default
`~/.config/aiswe/generation-mode.json`) written by
`scripts/classroom_gpu_mode.sh` with an atomic rename, and read here before
every generation (a `stat`, re-parsed only when it changed). Switching is
therefore one file write: no service restart, nothing in flight dropped, and
the VM services stay up the whole time so switching back is immediate.

A missing file means vm. A file that cannot be read or parsed also means vm,
loudly logged: the VM path is the one that is always running, and a typo must
not take a class offline. The GPU URL must be a loopback address, because the
file decides where students' questions are sent.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

MODE_FILE_ENV = "GENERATION_MODE_FILE"
DEFAULT_MODE_FILE = Path.home() / ".config" / "aiswe" / "generation-mode.json"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

VM = "vm"
GPU = "gpu"


@dataclass(frozen=True)
class GenerationMode:
    kind: str = VM
    gpu_url: str | None = None
    #: Informational, written by the switch script: when and onto what.
    details: dict[str, Any] = field(default_factory=dict)
    #: Why a mode file was ignored, when it was.
    error: str | None = None

    @property
    def is_gpu(self) -> bool:
        return self.kind == GPU


VM_MODE = GenerationMode()


def mode_file_path() -> Path:
    """The configured file; under test only an explicitly configured one.

    A developer's own `~/.config/aiswe/generation-mode.json` must not send a
    test run's requests to a GPU, so test mode never falls back to it.
    """
    raw = (os.getenv(MODE_FILE_ENV) or "").strip()
    if raw:
        return Path(raw).expanduser()
    from app.config import is_test_mode

    return Path(os.devnull + ".absent-generation-mode") if is_test_mode() else DEFAULT_MODE_FILE


def validate_gpu_url(raw: Any) -> str:
    """`http://127.0.0.1:<port>` and nothing else: the tunnel's local end."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("gpuUrl is required in gpu mode.")
    url = raw.strip().rstrip("/")
    parts = urlsplit(url)
    if parts.scheme != "http" or parts.hostname not in LOOPBACK_HOSTS or not parts.port:
        raise ValueError(f"gpuUrl must be http://127.0.0.1:<port>, got {raw!r}.")
    if parts.path or parts.query or parts.fragment:
        raise ValueError(f"gpuUrl must be an origin only, got {raw!r}.")
    return url


def parse_mode(text: str) -> GenerationMode:
    """A mode file's contents. Raises ValueError for anything not clearly valid."""
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"not JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    kind = data.get("mode")
    if kind == VM:
        return GenerationMode(kind=VM, details=_details(data))
    if kind == GPU:
        return GenerationMode(kind=GPU, gpu_url=validate_gpu_url(data.get("gpuUrl")), details=_details(data))
    raise ValueError(f"mode must be 'vm' or 'gpu', got {kind!r}")


def _details(data: dict[str, Any]) -> dict[str, Any]:
    keys = ("since", "node", "jobId", "expiresAt", "courseId", "modelVersion", "switchedBy")
    return {key: data[key] for key in keys if isinstance(data.get(key), (str, int, float))}


_lock = threading.Lock()
_cache: tuple[tuple[str, int, int] | None, GenerationMode] = (None, VM_MODE)
_listeners: list = []


def on_mode_change(listener) -> None:
    """Call `listener(old, new)` whenever the effective mode changes kind."""
    _listeners.append(listener)


def current_mode() -> GenerationMode:
    """The mode in effect now. Cheap enough to call before every generation."""
    global _cache
    path = mode_file_path()
    try:
        stat = path.stat()
        key: tuple[str, int, int] | None = (str(path), stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        key = None
    except OSError as exc:
        key = ("error", 0, 0)
        mode = GenerationMode(error=f"cannot stat {path}: {exc}")
        return _remember(key, mode)

    with _lock:
        if _cache[0] == key:
            return _cache[1]
    if key is None:
        return _remember(None, VM_MODE)
    try:
        mode = parse_mode(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        mode = GenerationMode(error=f"ignored {path}: {exc}")
    return _remember(key, mode)


def _remember(key, mode: GenerationMode) -> GenerationMode:
    global _cache
    with _lock:
        previous = _cache[1]
        _cache = (key, mode)
    if mode.error and mode.error != previous.error:
        logger.error("generation mode: %s; generating on the VM", mode.error)
    if mode.kind != previous.kind:
        logger.warning(
            "generation mode: %s -> %s%s",
            previous.kind,
            mode.kind,
            f" ({mode.gpu_url})" if mode.gpu_url else "",
        )
        for listener in list(_listeners):
            try:
                listener(previous, mode)
            except Exception:  # noqa: BLE001 - a listener must not break generation
                logger.exception("generation mode listener failed")
    return mode


def describe(mode: GenerationMode, *, ollama_url: str, ollama_model: str, finetuned_url: str | None) -> dict[str, Any]:
    """The admin view: which mode, and where each condition is generated."""
    if mode.is_gpu:
        gpu = f"Tillicum GPU via {mode.gpu_url}"
        routes = {
            "base": f"{gpu} (base model, adapters off)",
            "rag": f"{gpu} (base model, adapters off); retrieval on this VM",
            "fineTuned": f"{gpu} (course adapter)",
            "fineTunedRag": f"{gpu} (course adapter); retrieval on this VM",
        }
    else:
        routes = {
            "base": f"VM Ollama {ollama_url} ({ollama_model})",
            "rag": f"VM Ollama {ollama_url} ({ollama_model}); retrieval on this VM",
            "fineTuned": f"VM fine-tuned service {finetuned_url or '(not configured)'}",
            "fineTunedRag": f"VM fine-tuned service {finetuned_url or '(not configured)'}; retrieval on this VM",
        }
    return {
        "mode": mode.kind,
        "gpuUrl": mode.gpu_url,
        "details": mode.details,
        "error": mode.error,
        "modeFile": str(mode_file_path()),
        "routes": routes,
    }


def reset_for_tests() -> None:
    global _cache
    with _lock:
        _cache = (None, VM_MODE)
