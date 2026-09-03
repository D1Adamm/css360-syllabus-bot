"""Browser-safe views of the seed export, split and snapshot responses.

Why this exists
---------------
`seed_export` and `seed_split` work in absolute `Path`s, because they are
writing files and a writer needs a real location. Their summaries then record
what they wrote — `exportPath`, `files.trainJsonl`, `manifest.sourceFile` — and
those summaries were handed straight to the browser through
`summary: dict[str, Any]` response models. `project_root()` is
`Path(__file__).resolve().parents[2]`, so on the developer's machine that string
is `/Users/<name>/css360-syllabus-bot/…` and on the VM it is
`/home/<account>/…`: an unauthenticated endpoint publishing the operator's
account name.

The frontend never wanted the absolute form. `api.seedReview.test.ts` and
`AdminTrainingPage.test.tsx` both fix `exportPath` as
`data/exports/<courseId>/approved-finetune.jsonl` — repository-relative. That
was the contract; the backend drifted from it by passing a `str(Path)` through
a free-form dict. This module restores it at the API boundary.

Removal versus relativization
-----------------------------
`provenance_privacy` *drops* paths, because a compute node's run directory has a
logical equivalent (`artifactRef`, `datasetRef`) that already answers the same
question. Here the path is the answer: an administrator exporting a dataset is
told where it landed, and `data/exports/<courseId>/approved-finetune.jsonl` says
that exactly, on any machine, without naming an account. So these are rewritten
rather than removed, and the Admin Training status message keeps working.

Internal behaviour is untouched. `export_approved_seeds` still writes
`approved-export-summary.json` with the absolute paths it always did, and
`prepare_training_split` still writes the same `manifest.json` the cluster
worker verifies against. Only what leaves the process through an HTTP response
is rewritten.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from app.seed_export import project_root

#: Where a repository-relative reference restarts when the value is not under
#: this checkout — a path produced on another machine, or under a test root.
EXPORTS_MARKER = "data/exports/"

#: The same marker `relative_training_output_ref` uses on the cluster side, so a
#: training output that reaches one of these summaries is reduced the way the
#: rest of the system already reduces it.
TRAINING_OUTPUTS_MARKER = "training_outputs/"

#: Matches a value that names an absolute filesystem location: POSIX, home
#: relative, Windows drive, or UNC.
_ABSOLUTE = re.compile(r"^(?:/|~/|[A-Za-z]:[\\/]|\\\\)")

#: Keys whose value is a filesystem location. Rewritten wherever they appear,
#: at any depth, so that a relative-looking value is normalised too and a key
#: added later to one of these summaries is covered without a code change here.
PATH_VALUED_KEYS = frozenset(
    {
        "exportPath",
        "localSnapshotPath",
        "snapshotPath",
        "latestSnapshotPath",
        "sourceFile",
        "trainFile",
        "validationFile",
        "manifestFile",
        "finetuneJsonl",
        "metadataJson",
        "summaryJson",
        "trainJsonl",
        "validationJsonl",
        "manifestJson",
        "sourceJsonl",
    }
)


def repository_relative_path(value: Any) -> str:
    """One filesystem location as a stable, machine-independent reference.

    Never returns an absolute path. In order of preference:

    1. A path inside this checkout becomes its repository-relative form —
       `data/exports/<courseId>/approved-finetune.jsonl`.
    2. A path from elsewhere that still contains `data/exports/` (or
       `training_outputs/`) restarts at that marker, which is the same rule the
       cluster side already applies to run directories.
    3. Anything else is reduced to its file name. A bare name locates nothing
       and identifies nobody, and it is the part an operator reading a status
       message was going to recognise anyway.

    A value that is already relative is returned normalised, not rewritten: this
    is the identity function for the `data/exports/...` strings the frontend
    tests have always expected.
    """
    text = str(value or "").replace("\\", "/").strip()
    if not text:
        return ""

    if not _ABSOLUTE.match(text):
        return text[2:] if text.startswith("./") else text

    for marker in (EXPORTS_MARKER, TRAINING_OUTPUTS_MARKER):
        marker_at = text.rfind(marker)
        if marker_at >= 0:
            tail = text[marker_at:]
            # `training_outputs/` is stripped along with everything before it,
            # matching `relative_training_output_ref`; `data/exports/` is part
            # of the reference the frontend displays and is kept.
            if marker == TRAINING_OUTPUTS_MARKER:
                return tail[len(marker) :]
            return tail

    # Only a rooted POSIX path can be resolved against this checkout. `~/…`,
    # `C:\…` and `\\host\…` are not relative to anything here, and passing
    # them to `Path.resolve()` would join them onto the working directory and
    # invent a reference that names a file nobody wrote.
    if text.startswith("/"):
        try:
            return Path(text).resolve().relative_to(project_root()).as_posix()
        except (ValueError, OSError):
            pass

    return PurePosixPath(text).name


def _rewrite(value: Any) -> Any:
    """Recursively rewrite filesystem locations. Everything else is untouched."""
    if isinstance(value, Mapping):
        return {
            key: (
                repository_relative_path(item)
                if key in PATH_VALUED_KEYS and isinstance(item, str)
                else _rewrite(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_rewrite(item) for item in value]
    if isinstance(value, str) and _ABSOLUTE.match(value.replace("\\", "/")):
        # The backstop for a key nobody listed. Without it, renaming a field or
        # adding one to a summary would reopen exactly this leak.
        return repository_relative_path(value)
    return value


def public_export_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """An export or split summary as the browser may read it."""
    return _rewrite(summary)


def public_export_status(status: Mapping[str, Any]) -> dict[str, Any]:
    """The approved-export status record as the browser may read it."""
    return _rewrite(status)


def public_snapshot_ref(path: Any) -> str | None:
    """A generation snapshot location as the browser may read it, or None."""
    if path is None:
        return None
    reference = repository_relative_path(path)
    return reference or None


def public_message(message: Any) -> str:
    """One error message with any absolute path reduced to a relative one.

    `load_approved_finetune_records` and `validate_finetune_jsonl` name the file
    they could not read, and those messages become 4xx bodies. The file is worth
    naming; the account that owns it is not.
    """
    text = str(message)

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        # Trailing punctuation belongs to the sentence, not to the path.
        trailing = ""
        while token and token[-1] in ".,;:!?":
            trailing = token[-1] + trailing
            token = token[:-1]
        if not token:
            return match.group(0)
        return repository_relative_path(token) + trailing

    # A path token starts the string or follows a delimiter, so that a relative
    # reference already in the message is left alone.
    return re.sub(
        r"(?<![^\s\"'(\[<=,])(?:[A-Za-z]:[\\/]|~/|/(?![\s/]))[^\s\"'\]>)]*",
        replace,
        text,
    )
