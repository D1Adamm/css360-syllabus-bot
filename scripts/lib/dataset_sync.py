"""Fetch one run's prepared dataset onto the cluster, safely and idempotently.

Stdlib only, and Python 3.9 compatible: this runs on a Tillicum login node.

What this replaces
------------------
The manual step. Before this, a queued run reached a worker that refused it
because `data/exports/<courseId>/` did not exist locally, and recovering meant
logging into the application VM, running `sync_training_data_to_tillicum.sh`,
answering a Duo prompt for the rsync, and coming back to re-run the worker. The
data now travels on the credential the worker already holds, in the same
direction as every other call it makes.

The three properties that matter
--------------------------------
**Skip when identical.** The backend reports one `datasetSha256` over the whole
file set. The same digest computed over what is already on disk means the
transfer has nothing to do — which is the common case for a retried run, and the
reason a re-run of the worker is cheap rather than a re-download.

**Never publish a partial file.** Everything lands in a staging directory beside
the export, is verified against its digest there, and only then replaces the
real files. A transfer interrupted at any point leaves the previous dataset
exactly as it was; there is no window in which `train.jsonl` is half a file.
`os.replace` is what makes the last step atomic on the same filesystem.

**Verify before trusting.** Each file is checked against the digest the
descriptor gave for it, and the JSONL line counts are checked against the counts
the backend reported. A file that arrives truncated fails here rather than
becoming a silently shorter training set — which is the failure mode that
matters, because a short dataset trains successfully and produces a worse model
with nothing in any log to say why.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

CHUNK_BYTES = 64 * 1024

TRAIN_FILENAME = "train.jsonl"
VALIDATION_FILENAME = "validation.jsonl"
MANIFEST_FILENAME = "manifest.json"

#: Names this module will write. A descriptor naming anything else is refused
#: rather than followed: the backend and the worker have to agree on the file
#: set, and a name arriving over the network must never become a path segment
#: on its own authority.
ALLOWED_FILENAMES = frozenset(
    {
        TRAIN_FILENAME,
        VALIDATION_FILENAME,
        MANIFEST_FILENAME,
        "approved-export-summary.json",
    }
)

STAGING_PREFIX = ".dataset-download"


class DatasetSyncError(Exception):
    """The dataset could not be fetched, verified, or installed."""


@dataclass(frozen=True)
class DatasetSyncResult:
    """What happened, in terms an operator's console output can use."""

    course_id: str
    run_id: str
    export_dir: Path
    #: True when local files already matched the descriptor exactly.
    skipped: bool
    downloaded: List[str] = field(default_factory=list)
    train_count: int = 0
    validation_count: int = 0
    dataset_sha256: str = ""

    def describe(self) -> str:
        if self.skipped:
            return (
                f"already present and matching ({self.train_count} train / "
                f"{self.validation_count} validation)"
            )
        return (
            f"downloaded {len(self.downloaded)} file(s) "
            f"({self.train_count} train / {self.validation_count} validation)"
        )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def dataset_digest(entries: Dict[str, str]) -> str:
    """One digest over `{name: sha256}`, matching the backend's computation.

    Sorted by name so both sides agree regardless of how the mapping was built.
    """
    lines = sorted("{0}:{1}".format(name, digest) for name, digest in entries.items())
    return sha256_bytes("\n".join(lines).encode("utf-8"))


def count_jsonl_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def validate_descriptor(descriptor: Any) -> Dict[str, Any]:
    """Check a dataset description before acting on any part of it.

    Strict about file names and digests because both become instructions: a
    name decides where bytes are written, and a digest decides whether they are
    accepted. A descriptor missing either is not partially usable.
    """
    if not isinstance(descriptor, dict):
        raise DatasetSyncError("The backend returned an unreadable dataset description.")

    files = descriptor.get("files")
    if not isinstance(files, list) or not files:
        raise DatasetSyncError("The dataset description lists no files.")

    cleaned: List[Dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            raise DatasetSyncError("A dataset file entry was not an object.")
        name = item.get("name")
        digest = item.get("sha256")
        if not isinstance(name, str) or name not in ALLOWED_FILENAMES:
            raise DatasetSyncError(
                "The dataset description names a file this worker will not "
                "write: {0!r}.".format(name)
            )
        if not isinstance(digest, str) or len(digest) != 64:
            raise DatasetSyncError(
                "The dataset description gives no usable checksum for "
                "{0!r}; refusing to install an unverifiable file.".format(name)
            )
        cleaned.append({"name": name, "sha256": digest, "bytes": item.get("bytes")})

    names = {item["name"] for item in cleaned}
    missing = [
        required
        for required in (TRAIN_FILENAME, VALIDATION_FILENAME, MANIFEST_FILENAME)
        if required not in names
    ]
    if missing:
        raise DatasetSyncError(
            "The prepared dataset is incomplete: {0} missing.".format(
                ", ".join(missing)
            )
        )

    return {**descriptor, "files": cleaned}


def local_file_digests(export_dir: Path, names: List[str]) -> Dict[str, str]:
    """SHA-256 of each named file that is already present."""
    digests: Dict[str, str] = {}
    for name in names:
        path = export_dir / name
        if path.is_file():
            digests[name] = sha256_file(path)
    return digests


def local_dataset_matches(export_dir: Path, descriptor: Dict[str, Any]) -> bool:
    """Whether what is on disk is byte-for-byte the dataset described.

    Compared as a whole rather than file by file: a directory where two files
    match and one does not is not a dataset that can be trained on, and treating
    it as a partial hit would mean deciding which of two disagreeing versions to
    keep.
    """
    expected = {item["name"]: item["sha256"] for item in descriptor["files"]}
    actual = local_file_digests(export_dir, list(expected))
    if set(actual) != set(expected):
        return False
    return all(actual[name] == digest for name, digest in expected.items())


def _clear_staging(staging: Path) -> None:
    shutil.rmtree(staging, ignore_errors=True)


def _verify_counts(staging: Path, descriptor: Dict[str, Any]) -> Dict[str, int]:
    """Count the staged JSONL files and check them against the descriptor.

    The digests already prove the bytes arrived intact, so this cannot fail for
    a healthy transfer. It is here for the case the digests cannot see: a
    descriptor whose counts disagree with its own files, which would mean the
    backend prepared and described two different things.
    """
    train_count = count_jsonl_records(staging / TRAIN_FILENAME)
    validation_count = count_jsonl_records(staging / VALIDATION_FILENAME)

    expected_train = descriptor.get("trainExamples")
    expected_validation = descriptor.get("validationExamples")

    if isinstance(expected_train, int) and expected_train != train_count:
        raise DatasetSyncError(
            "The downloaded training set has {0} examples but the backend "
            "described {1}. Nothing was installed.".format(train_count, expected_train)
        )
    if isinstance(expected_validation, int) and expected_validation != validation_count:
        raise DatasetSyncError(
            "The downloaded validation set has {0} examples but the backend "
            "described {1}. Nothing was installed.".format(
                validation_count, expected_validation
            )
        )
    if train_count < 1:
        raise DatasetSyncError("The downloaded training set is empty.")

    return {"train_count": train_count, "validation_count": validation_count}


def install_from_staging(staging: Path, export_dir: Path, names: List[str]) -> None:
    """Move verified files into place, one atomic replace each.

    `os.replace` within the same filesystem is atomic, so no reader ever sees a
    half-written file. The set is not atomic as a whole — a crash between two
    replaces leaves one new file and one old one — which is why the caller
    re-verifies against the digests afterwards and why the descriptor comparison
    at the start of the next run treats a mismatched set as "fetch again".
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        os.replace(str(staging / name), str(export_dir / name))


def sync_run_dataset(
    queue: Any,
    course_id: str,
    run_id: str,
    export_dir: Path,
    *,
    descriptor: Optional[Dict[str, Any]] = None,
) -> DatasetSyncResult:
    """Make `export_dir` hold exactly the dataset the backend has for this run.

    `queue` is a `TrainingQueue`; only `describe_dataset` and
    `download_dataset_file` are used, so tests pass a stub rather than reaching
    the network.
    """
    described = validate_descriptor(
        descriptor
        if descriptor is not None
        else queue.describe_dataset(course_id, run_id)
    )
    names = [item["name"] for item in described["files"]]
    expected_digest = described.get("datasetSha256") or dataset_digest(
        {item["name"]: item["sha256"] for item in described["files"]}
    )

    if export_dir.is_dir() and local_dataset_matches(export_dir, described):
        counts = {
            "train_count": count_jsonl_records(export_dir / TRAIN_FILENAME),
            "validation_count": count_jsonl_records(export_dir / VALIDATION_FILENAME),
        }
        return DatasetSyncResult(
            course_id=course_id,
            run_id=run_id,
            export_dir=export_dir,
            skipped=True,
            downloaded=[],
            train_count=counts["train_count"],
            validation_count=counts["validation_count"],
            dataset_sha256=expected_digest,
        )

    staging = export_dir.parent / "{0}-{1}.{2}".format(
        STAGING_PREFIX, export_dir.name, os.getpid()
    )
    # A staging directory from a previous attempt that died is not a resource to
    # resume from — its contents were never verified. Start clean every time.
    _clear_staging(staging)
    staging.mkdir(parents=True, exist_ok=True)

    try:
        for item in described["files"]:
            name = item["name"]
            payload = queue.download_dataset_file(course_id, run_id, name)
            actual = sha256_bytes(payload)
            if actual != item["sha256"]:
                raise DatasetSyncError(
                    "Checksum mismatch for {0}: expected {1}, got {2}. Nothing "
                    "was installed.".format(name, item["sha256"][:12], actual[:12])
                )
            (staging / name).write_bytes(payload)

        counts = _verify_counts(staging, described)
        install_from_staging(staging, export_dir, names)
    except OSError as exc:
        # A full filesystem, a permission problem, or a transport that raised an
        # OSError rather than a queue error. All of them mean the same thing to
        # the caller — this dataset is not installed — and the `finally` below
        # has already ensured nothing partial survives.
        raise DatasetSyncError(
            "Could not complete the dataset transfer: {0}".format(exc)
        ) from exc
    finally:
        # One `finally` rather than a clean-up in each failure branch: whatever
        # ends this block — a checksum refusal, a full filesystem, a killed
        # session — must not leave unverified bytes sitting next to the real
        # export where the next run might mistake them for something.
        _clear_staging(staging)

    # Re-read from the installed location. The digests were verified in staging;
    # this proves the files that are actually going to be trained on are the
    # ones that were verified, including through the non-atomic multi-file move.
    if not local_dataset_matches(export_dir, described):
        raise DatasetSyncError(
            "The installed dataset does not match what was verified. The export "
            "directory may have been written by something else at the same "
            "time; re-run the worker."
        )

    return DatasetSyncResult(
        course_id=course_id,
        run_id=run_id,
        export_dir=export_dir,
        skipped=False,
        downloaded=names,
        train_count=counts["train_count"],
        validation_count=counts["validation_count"],
        dataset_sha256=expected_digest,
    )


def read_manifest_checksums(export_dir: Path) -> Dict[str, str]:
    """The `checksums` block a prepared manifest carries, or an empty mapping.

    Manifests written before checksums existed simply do not have the key. That
    is not an error: the descriptor is the authority during transfer, and this
    is for reporting what the dataset says about itself afterwards.
    """
    path = export_dir / MANIFEST_FILENAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    checksums = payload.get("checksums")
    if not isinstance(checksums, dict):
        return {}
    return {
        str(name): str(digest)
        for name, digest in checksums.items()
        if isinstance(digest, str)
    }
