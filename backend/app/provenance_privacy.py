"""Browser-safe views of training provenance.

Why this exists
---------------
`course_model_versions.provenance` and `training_runs.completion` are written
by the cluster, from files a job left on disk. Two of those files —
`resolved_config.json` and anything a traceback ended up inside — hold the
*runtime* location of the run: `/gpfs/projects/simswe/<netid>/training_outputs/
…`, `/home/<netid>/…`, `/Users/<name>/…`. That is genuinely useful when an
operator is reproducing a run on the cluster, and it is exactly the wrong thing
to hand a browser: every `/api/db` route is reachable without a credential, so
an absolute run path publishes the operator's UW NetID to anyone who can name a
course.

The record itself is not the problem, so the record is not what changes. This
module is a read-side view, applied where the response leaves the process, in
the same spirit as `db_serving_sessions.public_serving_session`, which drops a
compute node and port for the same reason. Stored provenance keeps the exact
paths for the debugging and reproducibility they were written for.

What is removed
---------------
1. Keys that hold a filesystem location and nothing else — `output_dir`,
   `train_path`, `validation_path` and their neighbours — are dropped from
   provenance wherever they are nested. They have logical equivalents the
   response already carries: `artifactRef`, `outputRef`, `datasetRef`,
   `datasetVersion`. A client that wants to know what a model was trained on
   reads those; nothing needs the compute node's directory layout.

2. Any remaining string that *is* an absolute path is dropped with its key, and
   an absolute path embedded in free text — a failure message, most often — is
   replaced by `[path removed]`, keeping the sentence that explains what went
   wrong. Blanket rather than "paths containing a username", because the
   provenance blob is open-ended: it is whatever a training script wrote, and a
   rule that only catches the two roots we know about today catches nothing the
   next cluster adds.

Nothing is rewritten into a substitute path. A path the client does not need is
removed, not disguised as `/gpfs/projects/simswe/<UW_NETID>/…`, which would
still describe a private filesystem while pretending not to.

What is deliberately left alone
-------------------------------
The version's own required fields — `artifactRef`, `baseModel`, `version` — are
untouched. `artifactRef` is already relative by construction: both writers
(`_validate_artifact_ref` on the completion path, `scripts/
register_course_model.py` on the manual one) refuse an absolute reference, so
there is nothing here to strip, and a view that could drop a required field
would turn a privacy fix into a 500.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

#: Replaces an absolute path found inside a longer string.
REDACTED = "[path removed]"

#: What a lease holder is called in a public response.
#:
#: `training_runs.claim_owner` is written by the worker as
#: `getpass.getuser()@socket.gethostname()` (`run_training_queue.py`
#: `default_owner`), so the stored value is a UW NetID and a machine name. It is
#: the right thing to store — an operator looking at the queue on the cluster
#: needs to know which of their sessions holds a run, and the worker-token
#: routes still report it exactly — and the wrong thing to publish, because
#: `/api/db` needs no credential.
#:
#: A constant rather than a hash or a truncation: the admin UI renders "Held by
#: {owner} until {time}", and the only fact that sentence needs is that a worker
#: holds the lease. Which worker is not a question the browser can act on — it
#: cannot reach the cluster, release the lease, or tell two runners apart — so
#: nothing is lost by saying so plainly.
PUBLIC_CLAIM_OWNER = "training-worker"

#: Keys whose value is a runtime filesystem location and nothing else. Dropped
#: wherever they appear inside provenance, at any nesting depth, because the
#: response already carries the logical reference that answers the same
#: question.
PATH_ONLY_KEYS = frozenset(
    {
        "output_dir",
        "outputDir",
        "train_path",
        "trainPath",
        "validation_path",
        "validationPath",
        "train_file",
        "trainFile",
        "validation_file",
        "validationFile",
        "source_file",
        "sourceFile",
        "export_path",
        "exportPath",
        "export_dir",
        "exportDir",
        "cache_dir",
        "cacheDir",
        "logging_dir",
        "loggingDir",
        "adapter_path",
        "adapterPath",
        "repo_root",
        "repoRoot",
    }
)

#: One absolute path token: POSIX (`/gpfs/…`), home-relative (`~/…`), Windows
#: (`C:\…`) or UNC (`\\host\…`).
#:
#: The lookbehind is what keeps a *relative* reference intact. Without it the
#: POSIX branch would match the `/adapter` inside
#: `qlora-runs/css-350/…-full/adapter`, and the safe logical references this
#: module exists to preserve would be the first casualty. A path token has to
#: start the string or follow a delimiter.
_PATH_TOKEN = re.compile(
    r"""
    (?<![^\s"'(\[<=,])
    (?:
        [A-Za-z]:[\\/]      # C:\ or C:/
      | \\\\[^\s\\]+        # \\host
      | ~/                  # ~/
      | /(?![\s/])          # POSIX absolute, but not a bare "/"
    )
    [^\s"'\]>)]*
    """,
    re.VERBOSE,
)

#: Trailing punctuation belongs to the sentence, not the path.
_TRAILING_PUNCTUATION = ".,;:!?"


class _Drop:
    """Sentinel: this value is a path and carries nothing else."""


DROP = _Drop()


def redact_paths(text: str) -> str:
    """Replace absolute path tokens in one string. Relative refs survive."""

    def replace(match: re.Match[str]) -> str:
        trailing = ""
        token = match.group(0)
        while token and token[-1] in _TRAILING_PUNCTUATION:
            trailing = token[-1] + trailing
            token = token[:-1]
        if not token:
            return match.group(0)
        return REDACTED + trailing

    return _PATH_TOKEN.sub(replace, text)


def _scrub(value: Any) -> Any:
    """Recursively remove filesystem locations. `DROP` means "omit this"."""
    if isinstance(value, str):
        redacted = redact_paths(value)
        if redacted.strip() == REDACTED:
            # The whole value was a path, so there is no sentence to keep.
            return DROP
        return redacted
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in PATH_ONLY_KEYS:
                continue
            scrubbed = _scrub(item)
            if scrubbed is DROP:
                continue
            cleaned[key] = scrubbed
        return cleaned
    if isinstance(value, (list, tuple)):
        items = [_scrub(item) for item in value]
        return [item for item in items if item is not DROP]
    return value


def public_provenance(provenance: Any) -> dict[str, Any] | None:
    """The browser-safe view of one provenance blob, or None."""
    if not isinstance(provenance, Mapping):
        return None
    return _scrub(provenance)


#: `completion` is the same kind of open-ended, cluster-written blob as
#: `provenance` — `resolvedConfig`, `runtimeReport` and a failure message all
#: live on it — so it gets the same treatment.
public_completion = public_provenance


def public_model_version(version: Mapping[str, Any]) -> dict[str, Any]:
    """One registered version as the browser may read it.

    Only `provenance` is rewritten. Everything else on the record is a contract
    field with its own validation, and `artifactRef` in particular is relative
    by construction on both write paths.
    """
    record = dict(version)
    if "provenance" in record:
        cleaned = public_provenance(record["provenance"])
        if cleaned is None:
            record.pop("provenance")
        else:
            record["provenance"] = cleaned
    return record


def public_model_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    """A course's model registry as the browser may read it."""
    record = dict(registry)
    versions = record.get("versions")
    if isinstance(versions, Mapping):
        record["versions"] = {
            key: public_model_version(value) if isinstance(value, Mapping) else value
            for key, value in versions.items()
        }
    return record


def public_training_run(run: Mapping[str, Any]) -> dict[str, Any]:
    """One queued run as the browser may read it.

    `completion` is the cluster's blob; `error` is free text a worker wrote, and
    a traceback that reached it can name the directory the job died in.
    """
    record = dict(run)

    claim = record.get("claim")
    if isinstance(claim, Mapping):
        # The lease itself is real and is reported as it stands — when it was
        # taken and when it lapses are what the queue is read for. Only the
        # account name is replaced, and the shape is preserved because
        # `parseTrainingRun` drops a claim that has no owner.
        record["claim"] = {**claim, "owner": PUBLIC_CLAIM_OWNER}

    if "completion" in record:
        cleaned = public_completion(record["completion"])
        if cleaned is None:
            record.pop("completion")
        else:
            record["completion"] = cleaned
    error = record.get("error")
    if isinstance(error, str):
        record["error"] = redact_paths(error)
    return record


#: Free-text fields on a model request that record why a stage failed. The
#: message is worth keeping; a path inside it is not.
_MODEL_REQUEST_ERROR_FIELDS = ("preparationError", "launchError")

#: Sub-objects an administrator's browser writes onto a model request.
_MODEL_REQUEST_DETAIL_FIELDS = ("preparation", "training")


def public_model_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """One model request as the browser may read it.

    `preparation` and `training` are written by the admin UI from fields it
    chose deliberately — `datasetRef` rather than a manifest path — so this is a
    guard rather than a fix for a known leak. It costs one pass and it means a
    future writer cannot quietly put a path on a record a professor reads.
    """
    record = dict(request)
    for field in _MODEL_REQUEST_DETAIL_FIELDS:
        value = record.get(field)
        if isinstance(value, Mapping):
            record[field] = _scrub(value)
    for field in _MODEL_REQUEST_ERROR_FIELDS:
        value = record.get(field)
        if isinstance(value, str):
            record[field] = redact_paths(value)
    return record
