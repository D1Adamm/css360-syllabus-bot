import { assertValidCourseId } from './courseId';
import { updateCourseModelRequest } from './courseModelRequestDb';
import {
  DuplicateTrainingRunError,
  enqueueTrainingRun,
  fetchCourseTrainingRuns,
  findActiveTrainingRun,
} from './trainingRunDb';
import type { CourseModelRequest, TrainingMode, TrainingRun } from '../types';

/**
 * Queueing a prepared request for training.
 *
 * The browser's durable responsibility ends here. It writes a `queued` run and
 * a pointer to it, and that is all: it does not reach a cluster, hold a
 * connection open, or wait for anything. Whoever runs the queue on the cluster
 * has logged in normally, with the usual two-factor prompt, and picks the work
 * up from Firebase.
 *
 * That is the whole reason this replaced the direct launch path. Submitting a
 * job from a web request needed a non-interactive session to a machine that
 * deliberately does not offer one, so the button could only ever be disabled.
 * A queue entry needs nothing interactive at all.
 *
 * The request's status is left at `preparing`. Nothing is training — a run is
 * waiting to be picked up — and telling a professor otherwise would be a lie
 * they read directly on their own page.
 */

export class TrainingNotPreparedError extends Error {
  constructor() {
    super('This request has no prepared training data yet.');
    this.name = 'TrainingNotPreparedError';
  }
}

/** Whether an admin should be offered a Queue training control. */
export function canQueueTraining(
  request: CourseModelRequest | null,
  runs: TrainingRun[] = [],
): boolean {
  if (!request) {
    return false;
  }

  return (
    request.status === 'preparing' &&
    Boolean(request.preparation) &&
    findActiveTrainingRun(runs) === null
  );
}

export interface QueueTrainingResult {
  run: TrainingRun;
}

export async function queueTrainingForRequest(
  courseId: string,
  request: CourseModelRequest | null,
  mode: TrainingMode = 'full',
): Promise<QueueTrainingResult> {
  assertValidCourseId(courseId);

  if (!request || request.status !== 'preparing' || !request.preparation) {
    throw new TrainingNotPreparedError();
  }

  // A cheap pre-check so the common duplicate is refused before a write is
  // attempted. It is not the guard — `enqueueTrainingRun` decides atomically,
  // because anything read here can be stale by the time the write lands.
  const existing = await fetchCourseTrainingRuns(courseId);
  if (findActiveTrainingRun(existing) !== null) {
    throw new DuplicateTrainingRunError();
  }

  const run = await enqueueTrainingRun(courseId, {
    mode,
    // The reference the preparation stage recorded, not anything absolute a
    // machine reported.
    datasetRef: request.preparation.datasetRef,
    approvedExampleCount: request.preparation.sourceApprovedExampleCount,
    trainExamples: request.preparation.trainExamples,
    validationExamples: request.preparation.validationExamples,
  });

  /*
   * Point the request at the run.
   *
   * A pointer and nothing more. The status stays `preparing` — the run is
   * queued, not training — and clearing `launchError` removes a message about
   * a route that no longer exists.
   */
  await updateCourseModelRequest(courseId, {
    currentRunId: run.runId,
    launchError: '',
  });

  return { run };
}
