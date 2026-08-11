import {
  exportApprovedCourseSeeds,
  listCourseSeeds,
  prepareTrainingSplit,
} from './api';
import { assertValidCourseId } from './courseId';
import { updateCourseModelRequest } from './courseModelRequestDb';
import { countExamples } from './exampleCounts';
import { RECOMMENDED_APPROVED_EXAMPLES } from './modelStatus';
import type { CourseModelRequestPreparation } from '../types';

/**
 * Turns a `requested` model request into a prepared training dataset.
 *
 * Deliberately no new backend code. Steps 3 and 4 — exporting approved examples
 * in the QLoRA fine-tune format and producing the deterministic train/validation
 * split — are exactly what `POST .../seeds/export-approved` and
 * `POST .../seeds/prepare-training-split` already do, writing into the
 * course-scoped `data/exports/{courseId}/`. Reimplementing either would give us
 * a second split that could disagree with the one training actually consumes.
 *
 * What this adds is the part that did not exist: re-checking the approved count
 * at preparation time, refusing when it is short, and recording the outcome on
 * the request.
 *
 * It stops at prepared data. No job is submitted, nothing is trained, no
 * adapter is promoted, and the model registry is not touched.
 *
 * Course isolation: `courseId` is validated once and then passed to every call.
 * Each endpoint scopes its own reads and writes by that id, so preparing one
 * course can never read or export another's examples.
 */

export class InsufficientApprovedExamplesError extends Error {
  readonly approved: number;
  readonly required: number;

  constructor(approved: number, required: number) {
    super(
      `This course has ${approved} approved example${approved === 1 ? '' : 's'}; ` +
        `${required} are needed before preparing training data.`,
    );
    this.name = 'InsufficientApprovedExamplesError';
    this.approved = approved;
    this.required = required;
  }
}

/**
 * Course-scoped, machine-independent dataset reference.
 *
 * The backend returns absolute paths; storing one would embed a machine layout
 * in a record the professor UI also reads.
 */
export function datasetRefForCourse(courseId: string): string {
  return `exports/${courseId}`;
}

export interface PrepareTrainingDataResult {
  preparation: CourseModelRequestPreparation;
}

export interface PrepareTrainingDataOptions {
  /** Override for tests. Defaults to the shared recommendation. */
  minimumApproved?: number;
}

export async function prepareTrainingDataForRequest(
  courseId: string,
  { minimumApproved = RECOMMENDED_APPROVED_EXAMPLES }: PrepareTrainingDataOptions = {},
): Promise<PrepareTrainingDataResult> {
  assertValidCourseId(courseId);

  try {
    // 1. Re-count now. The professor's original figure was recorded when they
    //    asked and may be stale — examples can be rejected after a request.
    const seeds = await listCourseSeeds(courseId);
    const counts = countExamples(seeds.seeds ?? []);

    if (counts.approved < minimumApproved) {
      throw new InsufficientApprovedExamplesError(counts.approved, minimumApproved);
    }

    // 2. Export approved-only examples for this course, in the format the
    //    fine-tune workflow already consumes.
    await exportApprovedCourseSeeds(courseId);

    // 3. Deterministic train/validation split over that export.
    const split = await prepareTrainingSplit(courseId);

    const preparation: CourseModelRequestPreparation = {
      preparedAt: new Date().toISOString(),
      sourceApprovedExampleCount: counts.approved,
      datasetRef: datasetRefForCourse(courseId),
      trainExamples: Number(split.summary?.trainExamples) || 0,
      validationExamples: Number(split.summary?.validationExamples) || 0,
      ...(Number.isFinite(Number(split.summary?.splitSeed))
        ? { splitSeed: Number(split.summary?.splitSeed) }
        : {}),
    };

    /*
     * Move to `preparing` and record what was produced.
     *
     * `preparing` — not `training`. Nothing is training: a dataset exists and a
     * job has not been submitted. Saying `training` here would be a lie that a
     * professor reads directly.
     */
    await updateCourseModelRequest(courseId, {
      status: 'preparing',
      preparation,
      // Clear any error from a previous attempt.
      preparationError: '',
    });

    return { preparation };
  } catch (error) {
    /*
     * Leave the request retryable rather than terminal.
     *
     * `failed` means the model could not be produced, and it is terminal — it
     * would unlock a fresh professor request over a hiccup in data preparation.
     * Returning it to `requested` with the reason recorded keeps the queue
     * honest and lets an administrator simply try again.
     */
    const message =
      error instanceof Error ? error.message : 'Preparation failed for an unknown reason.';

    try {
      await updateCourseModelRequest(courseId, {
        status: 'requested',
        preparationError: message,
      });
    } catch {
      // Reporting the original failure matters more than recording it.
    }

    throw error;
  }
}
