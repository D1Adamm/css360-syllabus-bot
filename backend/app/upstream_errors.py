"""One rule for what a failing upstream is allowed to tell the browser.

Every service this backend calls out to — the fine-tuned inference service on
Tillicum, the local Ollama server — answers a failure with a body, and that body
is written by whatever happened to answer: the service itself, an SSH tunnel, a
proxy in front of either, or the HTTP stack underneath. It routinely names a
host and port, a model cache under `~/.ollama/models`, a compute node, or a
serving root under `/gpfs`.

That makes it the useful half of the diagnostic and the unsafe half of the
answer. The public routes that surface these failures need no credential, so the
split is:

    upstream body  ->  the backend log, in full
    HTTP response  ->  what operation failed, and the status code

The status code is the part a client can act on; the body is the part an
operator reads. Neither loses anything by being where it belongs.

The caller passes its own logger so the record still says which subsystem
failed — `app.ollama` and `app.finetuned_client` are different problems with
different fixes, and an operator filtering the log should not have to guess.
"""

from __future__ import annotations

import logging
from typing import Any

#: How much of an upstream body to keep in the log. Generous on purpose: a log
#: line is the one place the whole thing is safe, and the old behaviour of
#: forwarding 200 characters to the browser was worse on both counts.
UPSTREAM_LOG_BODY_LIMIT = 2000


def log_upstream_failure(
    logger: logging.Logger,
    action: str,
    *,
    url: str,
    status_code: int,
    body: Any,
) -> None:
    """Record one upstream failure server-side, with its body and its URL."""
    text = body if isinstance(body, str) else str(body)
    logger.warning(
        "Upstream %s failed: HTTP %s from %s: %s",
        action,
        status_code,
        url,
        text[:UPSTREAM_LOG_BODY_LIMIT],
    )
