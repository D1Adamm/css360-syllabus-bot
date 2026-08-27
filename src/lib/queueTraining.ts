import { assertValidCourseId } from './courseId';
import { updateCourseModelRequest } from './courseModelRequestDb';
import { ApiError, enqueueTrainingRun, retryTrainingRun } from './api';
import {
  DuplicateTrainingRunError,
  fetchCourseTrainingRuns,
  findActiveTrainingRun,
} from './trainingRunDb';
import type { TrainingRun as TrainingRunType } from '../types';
import type { CourseModelRequest, TrainingMode, TrainingRun } from '../types';

/**
 * Queueing a prepared request for training.
 *
 * The browser's durable responsibility ends here. It asks the backend to write
 * a `queued` run and a pointer to it, and that is all: it does not reach a
 * cluster, hold a connection open, or wait for anything. Whoever runs the queue
 * on the cluster has logged in normally, with the usual two-factor prompt, and
 * claims the work from the same `training_runs` table.
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
  // attempted. It is not the guard — the backend's conditional INSERT decides
  // atomically, because anything read here can be stale by the time the write
  // lands.
  const existing = await fetchCourseTrainingRuns(courseId);
  if (findActiveTrainingRun(existing) !== null) {
    throw new DuplicateTrainingRunError();
  }

  let queued;
  try {
    queued = await enqueueTrainingRun(courseId, {
      mode,
      // The reference the preparation stage recorded, not anything absolute a
      // machine reported.
      datasetRef: request.preparation.datasetRef,
      approvedExampleCount: request.preparation.sourceApprovedExampleCount,
      trainExamples: request.preparation.trainExamples,
      validationExamples: request.preparation.validationExamples,
    });
  } catch (error) {
    // The backend refuses a second outstanding run with 409, the same rule the
    // browser used to enforce itself.
    if (error instanceof ApiError && error.status === 409) {
      throw new DuplicateTrainingRunError();
    }
    throw error;
  }

  const run = { ...(queued.run as object), runId: queued.runId } as TrainingRunType;

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


/* ------------------------------------------------------------------------ *
 * Retrying a stale run
 *
 * A run reaches `submitted`, the cluster finishes the job, and no completion
 * callback ever lands. The request still says `training` and the run still
 * says `submitted`, so `canQueueTraining` refuses a replacement — correctly,
 * because outstanding work is exactly what it is there to block. Retry is the
 * way out: it retires the stale run and queues a fresh one against the same
 * prepared dataset.
 *
 * The decision below is advisory. It decides whether to *offer* the control;
 * the backend decides, under a row lock, whether the action happens. The two
 * agree deliberately, and where they disagree the backend wins.
 * ------------------------------------------------------------------------ */

/** Terminal, so nothing is running and nothing can report. */
const RETRYABLE_RUN_STATES: readonly TrainingRun['state'][] = ['failed'];

/**
 * States where a cluster job may still exist.
 *
 * Neither the browser nor the backend can see Slurm, so neither can tell a
 * lost callback from a running job. Silence is the only evidence there is:
 * a healthy job's own callbacks are what would have moved `updatedAt`.
 */
const LIVE_JOB_RUN_STATES: readonly TrainingRun['state'][] = [
  'submitted',
  'training',
];

/** Six hours, matching `SUBMITTED_STALE_AFTER_SECONDS` on the backend. */
const SUBMITTED_STALE_AFTER_MS = 6 * 60 * 60 * 1000;

/** True when nothing can be said to hold this run any more. */
function claimHasExpired(run: TrainingRun, now: Date): boolean {
  if (!run.claim) {
    return true;
  }
  const expiresAt = Date.parse(run.claim.expiresAt);
  return Number.isNaN(expiresAt) || expiresAt <= now.getTime();
}

/**
 * The run a retry would act on: the one the request points at.
 *
 * Falls back to the newest run only when the request carries no pointer, which
 * is the shape of a request queued before `currentRunId` was written.
 */
export function findRetryTargetRun(
  request: CourseModelRequest | null,
  runs: TrainingRun[] = [],
): TrainingRun | null {
  if (!request) {
    return null;
  }
  if (request.currentRunId) {
    return runs.find((run) => run.runId === request.currentRunId) ?? null;
  }
  return runs.length > 0 ? (runs[runs.length - 1] ?? null) : null;
}

/**
 * Whether an admin should be offered a Retry training control.
 *
 * Conservative about `queued` and `claimed` for the same reason the backend is:
 * a queued run is already what a retry would produce, and a claimed run is
 * being worked on right now. An expired lease is the one claimed case that can
 * be *proved* stale, and it is the same evidence a worker uses to retake work.
 */
export function canRetryTraining(
  request: CourseModelRequest | null,
  runs: TrainingRun[] = [],
  now: Date = new Date(),
): boolean {
  const run = findRetryTargetRun(request, runs);
  if (!request || !run) {
    return false;
  }

  // Nothing to point a replacement job at.
  if (!run.datasetRef && !request.preparation?.datasetRef) {
    return false;
  }

  if (RETRYABLE_RUN_STATES.includes(run.state)) {
    return true;
  }

  if (LIVE_JOB_RUN_STATES.includes(run.state)) {
    const updatedAt = Date.parse(run.updatedAt || run.enqueuedAt);
    // An unreadable timestamp is treated as stale, matching the backend: a run
    // nothing can date is one nothing has reported against in a readable way.
    return (
      Number.isNaN(updatedAt) ||
      now.getTime() - updatedAt >= SUBMITTED_STALE_AFTER_MS
    );
  }

  return run.state === 'claimed' && claimHasExpired(run, now);
}

export interface RetryTrainingResult {
  run: TrainingRun;
  supersededRunId: string;
}

/**
 * Ask the backend to retire the current run and queue its replacement.
 *
 * One call, one transaction. Nothing about which run is retired or what the
 * replacement inherits is computed here — sending that from a browser would be
 * sending state that was already stale when it was read.
 */
export async function retryTrainingForRequest(
  courseId: string,
): Promise<RetryTrainingResult> {
  assertValidCourseId(courseId);

  const response = await retryTrainingRun(courseId);
  const run = {
    ...(response.run as object),
    runId: response.runId,
  } as TrainingRunType;

  return { run, supersededRunId: response.supersededRunId };
}
