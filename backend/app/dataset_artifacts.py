"""The prepared training dataset for one course, described and served safely.

Why this exists
---------------
The dataset was previously moved to the cluster by hand: prepare on the VM,
`rsync` the export directory over SSH, answer a Duo prompt, then go back to
Tillicum and re-run the worker, which had refused the run because
`data/exports/<courseId>/` was not there yet. Six steps and two authentications
between queueing a run and training it, every time.

The worker is already an authenticated caller of this backend — that is how it
claims runs and reports submissions. So the dataset travels the same way: one
outbound HTTPS request from the cluster, on the credential it already holds.

What a caller is allowed to ask for
-----------------------------------
Not a path. The only inputs are a course id and a run id, both already validated
and already meaningful to the queue; this module turns them into a directory.
File names are matched against `DATASET_FILES`, a fixed tuple — a name that is
not in it is refused before any path is built, so `..`, an absolute path and a
symlink target are all simply names that do not match. There is no code here
that joins caller-supplied text onto a directory.

The run is what scopes the request. `resolve_run_dataset` takes the run's own
`datasetRef` and checks it against the reference this course's export directory
would produce, so a run cannot be pointed at another course's data even if its
stored reference said so.

Integrity
---------
Every file is described with its size and its SHA-256, and the set of them is
summarised into one `datasetSha256`. That single value is what lets the worker
decide it already has exactly this dataset and skip the transfer entirely; the
per-file digests are what it verifies each downloaded file against before
allowing it to replace anything.

`prepare_training_split` now writes the train/validation digests into
`manifest.json` as well, so the manifest is self-describing for anyone reading
it later on the cluster. The manifest cannot contain its own digest, which is
why the descriptor carries `manifestSha256` separately.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.course_id import assert_valid_course_id
from app.seed_export import course_export_dir

TRAIN_FILENAME = "train.jsonl"
VALIDATION_FILENAME = "validation.jsonl"
MANIFEST_FILENAME = "manifest.json"
EXPORT_SUMMARY_FILENAME = "approved-export-summary.json"

#: Exactly what a worker may fetch, and whether training can start without it.
#:
#: A fixed tuple rather than a directory listing: a listing would hand out
#: whatever happened to be in the export directory, including files a later
#: feature writes there for a different audience. Adding a file to the transfer
#: is a deliberate edit here.
#:
#: `approved-export-summary.json` is optional and carries the export timestamp
#: and approved count the manifest refers to. It is included so a run can be
#: traced back to the review state it came from without a database read on the
#: cluster; its absence is not a reason to refuse to train.
DATASET_FILES: tuple[tuple[str, bool], ...] = (
    (TRAIN_FILENAME, True),
    (VALIDATION_FILENAME, True),
    (MANIFEST_FILENAME, True),
    (EXPORT_SUMMARY_FILENAME, False),
)

REQUIRED_DATASET_FILES = tuple(name for name, required in DATASET_FILES if required)
DATASET_FILE_NAMES = tuple(name for name, _ in DATASET_FILES)

#: Refuse to describe or serve anything larger. The prepared export for a course
#: this size is a few hundred kilobytes; a file in the tens of megabytes means
#: something other than a split dataset is in the directory, and streaming it to
#: the cluster is not a thing this endpoint should do on a guess.
MAX_DATASET_FILE_BYTES = 64 * 1024 * 1024

CHUNK_BYTES = 64 * 1024


class DatasetArtifactError(Exception):
    """The prepared dataset for a run is missing or unusable."""


class UnknownDatasetFileError(DatasetArtifactError):
    """A name that is not one of this dataset's files was requested."""


def dataset_ref_for_course(course_id: str) -> str:
    """The relative reference a prepared export is recorded under.

    Matches what `prepareTrainingData` writes onto the model request and what
    the enqueue route copies onto the run, so a stored `datasetRef` can be
    compared against the course it claims to belong to.
    """
    return f"exports/{assert_valid_course_id(course_id)}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dataset_digest(files: list[dict[str, Any]]) -> str:
    """One digest over the whole file set, order-independent.

    Built from `name:sha256` lines sorted by name, so the same dataset always
    produces the same value regardless of how the list was assembled. The worker
    compares this against the same computation over what it already has on disk;
    equal means there is nothing to transfer.
    """
    lines = sorted(f"{item['name']}:{item['sha256']}" for item in files)
    return sha256_text("\n".join(lines))


def count_jsonl_records(path: Path) -> int:
    """Non-blank lines in a JSONL file.

    The counts also live in the manifest and on the run row. This one is read
    off the bytes that will actually be transferred, which is the number worth
    comparing the other two against.
    """
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def resolve_dataset_dir(course_id: str, *, export_root: Path | None = None) -> Path:
    """This course's prepared export directory, or raise.

    The directory is derived from the validated course id alone. Nothing a
    caller sends contributes a path segment.
    """
    safe_course_id = assert_valid_course_id(course_id)
    directory = course_export_dir(safe_course_id, root=export_root)
    if not directory.is_dir():
        raise DatasetArtifactError(
            f'No prepared training data for "{safe_course_id}". Export the '
            "approved examples and prepare the train/validation split before "
            "a worker can fetch it."
        )
    return directory


def resolve_dataset_file(
    course_id: str,
    name: str,
    *,
    export_root: Path | None = None,
) -> Path:
    """One named file from this course's export, or raise.

    `name` is compared against the allowlist, never joined onto anything until
    it has matched one of its entries. A traversal attempt is therefore an
    unknown name and nothing more.
    """
    if name not in DATASET_FILE_NAMES:
        raise UnknownDatasetFileError(
            f"{name!r} is not part of a prepared training dataset. "
            f"Expected one of {list(DATASET_FILE_NAMES)}."
        )

    directory = resolve_dataset_dir(course_id, export_root=export_root)
    path = directory / name
    if not path.is_file():
        raise DatasetArtifactError(f'"{name}" has not been prepared for this course.')
    if path.stat().st_size > MAX_DATASET_FILE_BYTES:
        raise DatasetArtifactError(
            f'"{name}" is larger than a prepared dataset file should be '
            f"({MAX_DATASET_FILE_BYTES} bytes)."
        )
    return path


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetArtifactError(f"manifest.json is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise DatasetArtifactError("manifest.json must be a JSON object.")
    return payload


def describe_dataset(
    course_id: str,
    *,
    export_root: Path | None = None,
) -> dict[str, Any]:
    """Everything a worker needs to decide whether to fetch, and to verify.

    Counts come from the files themselves rather than from the manifest, so a
    manifest that has drifted from the data beside it cannot make a short
    dataset look complete. The manifest's own counts are reported alongside, so
    the worker can say which two disagree instead of just refusing.
    """
    safe_course_id = assert_valid_course_id(course_id)
    directory = resolve_dataset_dir(safe_course_id, export_root=export_root)

    missing = [
        name
        for name in REQUIRED_DATASET_FILES
        if not (directory / name).is_file()
    ]
    if missing:
        raise DatasetArtifactError(
            f'The prepared dataset for "{safe_course_id}" is incomplete: '
            f"{', '.join(missing)} missing. Re-run the train/validation split."
        )

    files: list[dict[str, Any]] = []
    for name, required in DATASET_FILES:
        path = directory / name
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_DATASET_FILE_BYTES:
            raise DatasetArtifactError(
                f'"{name}" is larger than a prepared dataset file should be '
                f"({MAX_DATASET_FILE_BYTES} bytes)."
            )
        files.append(
            {
                "name": name,
                "bytes": size,
                "sha256": sha256_file(path),
                "required": required,
            }
        )

    manifest = read_manifest(directory / MANIFEST_FILENAME)
    manifest_sha256 = next(
        item["sha256"] for item in files if item["name"] == MANIFEST_FILENAME
    )

    train_count = count_jsonl_records(directory / TRAIN_FILENAME)
    validation_count = count_jsonl_records(directory / VALIDATION_FILENAME)

    return {
        "courseId": safe_course_id,
        "datasetRef": dataset_ref_for_course(safe_course_id),
        "datasetVersion": manifest.get("datasetVersion"),
        "splitSeed": manifest.get("splitSeed"),
        "trainExamples": train_count,
        "validationExamples": validation_count,
        "totalExamples": train_count + validation_count,
        "manifestTrainExamples": manifest.get("trainExamples"),
        "manifestValidationExamples": manifest.get("validationExamples"),
        "manifestSha256": manifest_sha256,
        "datasetSha256": dataset_digest(files),
        "files": files,
    }


def resolve_run_dataset(
    run: dict[str, Any],
    *,
    export_root: Path | None = None,
) -> dict[str, Any]:
    """Describe the dataset belonging to one queued run.

    The run supplies the course, and the course supplies the directory. The
    run's stored `datasetRef` is checked rather than followed: it is metadata
    recorded when the data was prepared, and if it names something other than
    this course's export then the two records disagree and a worker must not be
    handed either of them silently.
    """
    course_id = assert_valid_course_id(str(run.get("courseId") or ""))
    expected_ref = dataset_ref_for_course(course_id)
    stored_ref = str(run.get("datasetRef") or "").strip()

    if stored_ref and stored_ref != expected_ref:
        raise DatasetArtifactError(
            f'This run records datasetRef "{stored_ref}", which is not this '
            f'course\'s prepared export ("{expected_ref}"). Nothing was served: '
            "re-prepare the training data for this course."
        )

    described = describe_dataset(course_id, export_root=export_root)
    described["runId"] = run.get("runId")
    described["approvedExampleCount"] = int(run.get("approvedExampleCount") or 0)
    described["enqueuedTrainExamples"] = int(run.get("trainExamples") or 0)
    described["enqueuedValidationExamples"] = int(run.get("validationExamples") or 0)
    return described
