"""Pin a training run to the prepared dataset as it is on disk right now.

Audit finding B10 (2026-10-27): a queued run named its dataset by directory
(`exports/<courseId>`) plus two counts, so the files could be re-prepared
underneath it and the worker would train on data nobody had queued. Every
route that creates a run — the browser's `POST /api/courses/{id}/training-runs`,
its `/api/db` twin, and the admin retry — now records the SHA-256 of the
prepared `train.jsonl` and `validation.jsonl` on the run, and
`dataset_artifacts.resolve_run_dataset` refuses to serve the worker files that
no longer match it.

One helper, used by every creating route, so the three cannot disagree about
what "pinned" means or which refusal a stale request gets.
"""

from __future__ import annotations

from fastapi import HTTPException

from app.dataset_artifacts import DatasetArtifactError, describe_dataset
from app.error_classification import (
    DATASET_CHANGED,
    DATASET_NOT_PREPARED,
    ClassifiedHTTPException,
)


def pin_dataset_for_enqueue(
    course_id: str,
    *,
    dataset_ref: str,
    train_examples: int,
    validation_examples: int,
) -> str:
    """The digest a run queued now will be pinned to, or an HTTP refusal.

    - no prepared files: `409 dataset_not_prepared` (prepare first);
    - a `datasetRef` that is not this course's export: 422, the request is
      malformed or for another course;
    - counts that disagree with the files: `409 dataset_changed` — the
      request's preparation record went stale (the data was re-prepared since)
      and the run must be queued from the current one.
    """
    try:
        described = describe_dataset(course_id)
    except DatasetArtifactError as exc:
        raise ClassifiedHTTPException(
            status_code=409, code=DATASET_NOT_PREPARED, message=str(exc)
        ) from exc
    if dataset_ref != described["datasetRef"]:
        raise HTTPException(
            status_code=422,
            detail=(
                f'datasetRef "{dataset_ref}" is not this course\'s prepared '
                f'export ("{described["datasetRef"]}").'
            ),
        )
    if (
        int(train_examples) != int(described["trainExamples"])
        or int(validation_examples) != int(described["validationExamples"])
    ):
        raise ClassifiedHTTPException(
            status_code=409,
            code=DATASET_CHANGED,
            message=(
                "The prepared training data on disk "
                f"({described['trainExamples']} train / {described['validationExamples']} "
                "validation) does not match what this request recorded "
                f"({train_examples} / {validation_examples}). "
                "Prepare the training data again, then queue the run."
            ),
        )
    return str(described["datasetSha256"])
