"""Backend configuration bootstrap.

Loads backend/.env once so uvicorn and tests can rely on process environment
variables without exporting them manually.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = BACKEND_ROOT / ".env"

_env_loaded_from: Path | None = None


def load_backend_env(
    env_file: Path | str | None = None,
    *,
    override: bool = False,
) -> bool:
    """Load environment variables from a .env file.

    Defaults to backend/.env. Returns True if the file was found and loaded.
    Existing process environment values win unless override=True.
    """
    global _env_loaded_from

    path = Path(env_file) if env_file is not None else DEFAULT_ENV_FILE
    loaded = load_dotenv(path, override=override)
    if loaded:
        _env_loaded_from = path
    return loaded


# Load backend/.env on import so os.getenv works for subsequent app modules.
load_backend_env()
