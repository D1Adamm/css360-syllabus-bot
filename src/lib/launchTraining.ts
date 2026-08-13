import { launchCourseTraining } from './adminApi';
import { assertValidCourseId } from './courseId';
import { updateCourseModelRequest } from './courseModelRequestDb';
import type { CourseModelRequest, CourseModelRequestTraining } from '../types';

/**
 * Submits a prepared course's training job and records the result.
 *
 * @deprecated Superseded by `queueTraining.ts`. Nothing in the application
 * calls this any more: an administrator enqueues a run at
 * `courses/{courseId}/trainingRuns` and a runner on the cluster picks it up.
 *
 * The direct path could never work from a browser. It needed the backend to
 * reach the cluster without a person present, and the cluster requires the
 * usual interactive two-factor login, so the control was permanently disabled
 * behind `TRAINING_LAUNCH_ENABLED`. That flag stays off.
 *
 * Kept rather than deleted because the module and its backend counterpart were
 * validated together, and removing working code that a future non-interactive
 * path might use is a larger change than leaving it unreferenced.
 *
 * The infrastructure itself lives behind the backend endpoint — nothing here
 * touches ssh, rsync, or sbatch, and the browser never learns a cluster host,
 * path, or token. This module owns only the request-state rules:
 *
 *   - launch only a request whose data has actually been prepared
 *   - never launch twice
 *   - move to `training` *after* a real Slurm job id comes back, never before
 *   - on failure, leave the request at `preparing` so it can be retried
 */

export class NotPreparedError extends Error {
  constructor() {
    super('This request has no prepared training data yet.');
    this.name = 'NotPreparedError';
  }
}

export class AlreadyLaunchedError extends Error {
  constructor(jobId: string) {
    super(`Training has already been submitted for this course (job ${jobId}).`);
    this.name = 'AlreadyLaunchedError';
  }
}

/** Whether an admin should be offered a Start training control. */
export function canLaunchTraining(request: CourseModelRequest | null): boolean {
  if (!request) {
    return false;
  }
  // Preparation must have produced a dataset, and no job may exist yet.
  return (
    request.status === 'preparing' &&
    Boolean(request.preparation) &&
    !request.training
  );
}

export interface LaunchTrainingResult {
  training: CourseModelRequestTraining;
}

export async function launchTrainingForRequest(
  courseId: string,
  request: CourseModelRequest | null,
  mode: 'smoke' | 'full' = 'full',
): Promise<LaunchTrainingResult> {
  assertValidCourseId(courseId);

  // Guard before the network call, so an ineligible request never reaches the
  // infrastructure boundary at all.
  if (request?.training?.jobId) {
    throw new AlreadyLaunchedError(request.training.jobId);
  }
  if (!request || request.status !== 'preparing' || !request.preparation) {
    throw new NotPreparedError();
  }

  try {
    const response = await launchCourseTraining(courseId, mode);

    const training: CourseModelRequestTraining = {
      jobId: response.jobId,
      mode: response.mode,
      submittedAt: response.submittedAt,
      // Keep the relative reference the preparation stage recorded rather than
      // anything absolute the cluster reported.
      datasetRef: request.preparation.datasetRef,
      trainExamples: response.trainCount,
      validationExamples: response.validationCount,
    };

    /*
     * Only now does the request become `training`.
     *
     * A real job id came back from a real submission. Setting this before the
     * call, or on a failure, would tell a professor their model was training
     * when nothing was queued.
     */
    await updateCourseModelRequest(courseId, {
      status: 'training',
      training,
      launchError: '',
    });

    return { training };
  } catch (error) {
    const message =
      error instanceof Error ? error.message : 'Launching training failed.';

    // Stay at `preparing`: the dataset is still valid and the launch can simply
    // be retried. Recording the reason keeps it visible to an administrator.
    try {
      await updateCourseModelRequest(courseId, {
        status: 'preparing',
        launchError: message,
      });
    } catch {
      // Reporting the original failure matters more than recording it.
    }

    throw error;
  }
}
