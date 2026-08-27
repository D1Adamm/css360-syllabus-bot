"""The prepared dataset a worker is allowed to fetch, and how it is described.

These cover the module that replaced the manual `rsync`. The properties worth
pinning are not "does it read a file" but the three that make an automatic
transfer safe to run unattended: a caller cannot name a path, a mismatched or
incomplete dataset is refused rather than half-served, and the digest a worker
compares against is computed the same way on both sides.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app import dataset_artifacts as artifacts

COURSE = "css-350-spring-2026-n3h9"
OTHER_COURSE = "css-360-winter-2026-a7rp"


def _jsonl(count: int, prefix: str) -> str:
    return "".join(
        json.dumps({"instruction": f"{prefix}{index}", "response": f"A{index}"}) + "\n"
        for index in range(count)
    )


def _prepare(
    root: Path,
    course_id: str = COURSE,
    *,
    train: int = 37,
    validation: int = 5,
    manifest: dict | None = None,
    summary: bool = False,
) -> Path:
    export_dir = root / "data" / "exports" / course_id
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "train.jsonl").write_text(_jsonl(train, "Q"), encoding="utf-8")
    (export_dir / "validation.jsonl").write_text(_jsonl(validation, "V"), encoding="utf-8")
    (export_dir / "manifest.json").write_text(
        json.dumps(
            manifest
            if manifest is not None
            else {
                "courseId": course_id,
                "datasetVersion": f"{course_id}-approved-split-seed360-n{train + validation}",
                "splitSeed": 360,
                "trainExamples": train,
                "validationExamples": validation,
            }
        ),
        encoding="utf-8",
    )
    if summary:
        (export_dir / "approved-export-summary.json").write_text(
            json.dumps({"exportedAt": "2026-08-27T06:00:00Z", "approvedCount": 42}),
            encoding="utf-8",
        )
    return export_dir


def _run(course_id: str = COURSE, **overrides) -> dict:
    return {
        "runId": "run-20260827t064701z-1cf650",
        "courseId": course_id,
        "datasetRef": f"exports/{course_id}",
        "approvedExampleCount": 42,
        "trainExamples": 37,
        "validationExamples": 5,
        **overrides,
    }


# --------------------------------------------------------------------------- #
# Describing
# --------------------------------------------------------------------------- #


def test_describe_reports_counts_digests_and_one_dataset_digest(tmp_path: Path) -> None:
    _prepare(tmp_path)

    described = artifacts.describe_dataset(COURSE, export_root=tmp_path)

    assert described["courseId"] == COURSE
    assert described["datasetRef"] == f"exports/{COURSE}"
    assert described["trainExamples"] == 37
    assert described["validationExamples"] == 5
    assert described["totalExamples"] == 42

    names = {item["name"] for item in described["files"]}
    assert names == {"train.jsonl", "validation.jsonl", "manifest.json"}
    assert all(len(item["sha256"]) == 64 for item in described["files"])
    assert len(described["datasetSha256"]) == 64


def test_counts_come_from_the_files_not_the_manifest(tmp_path: Path) -> None:
    """A manifest that has drifted cannot make a short dataset look complete.

    The manifest is written by the split and the files are written beside it, so
    they normally agree. When they do not, the number that decides whether a run
    is trainable has to be the one read off the bytes that will be transferred.
    """
    _prepare(
        tmp_path,
        train=37,
        validation=5,
        manifest={"courseId": COURSE, "trainExamples": 999, "validationExamples": 999},
    )

    described = artifacts.describe_dataset(COURSE, export_root=tmp_path)

    assert described["trainExamples"] == 37
    assert described["manifestTrainExamples"] == 999


def test_the_optional_summary_is_included_when_present(tmp_path: Path) -> None:
    _prepare(tmp_path, summary=True)

    described = artifacts.describe_dataset(COURSE, export_root=tmp_path)
    names = {item["name"] for item in described["files"]}

    assert "approved-export-summary.json" in names
    optional = next(
        item for item in described["files"] if item["name"] == "approved-export-summary.json"
    )
    assert optional["required"] is False


def test_a_missing_required_file_is_refused(tmp_path: Path) -> None:
    export_dir = _prepare(tmp_path)
    (export_dir / "validation.jsonl").unlink()

    with pytest.raises(artifacts.DatasetArtifactError) as excinfo:
        artifacts.describe_dataset(COURSE, export_root=tmp_path)

    assert "validation.jsonl" in str(excinfo.value)


def test_an_unprepared_course_says_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(artifacts.DatasetArtifactError) as excinfo:
        artifacts.describe_dataset(COURSE, export_root=tmp_path)

    assert "Export the approved examples" in str(excinfo.value)


def test_the_digest_matches_the_worker_side_computation(tmp_path: Path) -> None:
    """Both sides must agree, or the skip-if-identical path never fires.

    Recomputed here the way `scripts/lib/dataset_sync.py` does, rather than by
    calling that module: the point is that two independent implementations
    produce the same value.
    """
    _prepare(tmp_path)
    described = artifacts.describe_dataset(COURSE, export_root=tmp_path)

    lines = sorted(
        f"{item['name']}:{item['sha256']}" for item in described["files"]
    )
    expected = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()

    assert described["datasetSha256"] == expected


# --------------------------------------------------------------------------- #
# What a caller may name
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    [
        "../../../etc/passwd",
        "..",
        "/etc/passwd",
        "train.jsonl/../../secrets",
        "approved-finetune.jsonl",
        "manifest.json.bak",
        "",
    ],
)
def test_only_allowlisted_names_resolve(tmp_path: Path, name: str) -> None:
    """Traversal is not blocked, it is unrepresentable.

    A name is compared against a fixed tuple before any path exists, so `..` is
    simply a name that is not in the list. There is no join of caller text onto
    a directory anywhere in this module.
    """
    _prepare(tmp_path)

    with pytest.raises(artifacts.UnknownDatasetFileError):
        artifacts.resolve_dataset_file(COURSE, name, export_root=tmp_path)


def test_an_allowlisted_name_resolves_inside_the_course_directory(tmp_path: Path) -> None:
    _prepare(tmp_path)

    path = artifacts.resolve_dataset_file(COURSE, "train.jsonl", export_root=tmp_path)

    assert path == tmp_path / "data" / "exports" / COURSE / "train.jsonl"


def test_an_invalid_course_id_never_reaches_a_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        artifacts.resolve_dataset_file(
            "../other", "train.jsonl", export_root=tmp_path
        )


def test_an_oversized_file_is_refused(tmp_path: Path, monkeypatch) -> None:
    _prepare(tmp_path)
    monkeypatch.setattr(artifacts, "MAX_DATASET_FILE_BYTES", 10)

    with pytest.raises(artifacts.DatasetArtifactError) as excinfo:
        artifacts.resolve_dataset_file("css-350-spring-2026-n3h9", "train.jsonl", export_root=tmp_path)

    assert "larger than" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Run scoping
# --------------------------------------------------------------------------- #


def test_a_run_resolves_its_own_course_dataset(tmp_path: Path) -> None:
    _prepare(tmp_path)

    described = artifacts.resolve_run_dataset(_run(), export_root=tmp_path)

    assert described["courseId"] == COURSE
    assert described["runId"] == "run-20260827t064701z-1cf650"
    assert described["approvedExampleCount"] == 42


def test_a_run_pointing_at_another_courses_export_is_refused(tmp_path: Path) -> None:
    """`datasetRef` semantics are preserved by checking them, not by following them.

    A run whose stored reference names another course means two records
    disagree. Serving either of them silently is how one course's data reaches
    another course's model.
    """
    _prepare(tmp_path)
    _prepare(tmp_path, OTHER_COURSE)

    run = _run(datasetRef=f"exports/{OTHER_COURSE}")

    with pytest.raises(artifacts.DatasetArtifactError) as excinfo:
        artifacts.resolve_run_dataset(run, export_root=tmp_path)

    assert OTHER_COURSE in str(excinfo.value)
    assert "re-prepare" in str(excinfo.value).lower()


def test_a_run_with_no_recorded_dataset_ref_still_resolves(tmp_path: Path) -> None:
    """Runs enqueued before `datasetRef` was recorded are not stranded."""
    _prepare(tmp_path)

    described = artifacts.resolve_run_dataset(_run(datasetRef=""), export_root=tmp_path)

    assert described["trainExamples"] == 37


def test_two_courses_get_two_different_datasets(tmp_path: Path) -> None:
    _prepare(tmp_path, COURSE, train=37, validation=5)
    _prepare(tmp_path, OTHER_COURSE, train=12, validation=2)

    first = artifacts.describe_dataset(COURSE, export_root=tmp_path)
    second = artifacts.describe_dataset(OTHER_COURSE, export_root=tmp_path)

    assert first["trainExamples"] == 37
    assert second["trainExamples"] == 12
    assert first["datasetSha256"] != second["datasetSha256"]
