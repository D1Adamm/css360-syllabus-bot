import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { CourseModelRequest, TrainingRun } from '../types';

const enqueueTrainingRun = vi.fn();
const fetchCourseTrainingRuns = vi.fn();
const updateCourseModelRequest = vi.fn();
const retryTrainingRun = vi.fn();

vi.mock('./trainingRunDb', async () => {
  const actual = await vi.importActual<typeof import('./trainingRunDb')>(
    './trainingRunDb',
  );
  return {
    ...actual,
    fetchCourseTrainingRuns: (...args: unknown[]) => fetchCourseTrainingRuns(...args),
  };
});

// Queueing goes through the backend, which inserts the run into the
// `training_runs` table the admin list reads and the cluster claims from.
vi.mock('./api', async () => {
  const actual = await vi.importActual<typeof import('./api')>('./api');
  return {
    ...actual,
    enqueueTrainingRun: (...args: unknown[]) => enqueueTrainingRun(...args),
    retryTrainingRun: (...args: unknown[]) => retryTrainingRun(...args),
  };
});

vi.mock('./courseModelRequestDb', () => ({
  updateCourseModelRequest: (...args: unknown[]) => updateCourseModelRequest(...args),
}));

import { DuplicateTrainingRunError } from './trainingRunDb';
import {
  canQueueTraining,
  canRetryTraining,
  findRetryTargetRun,
  queueTrainingForRequest,
  retryTrainingForRequest,
  TrainingNotPreparedError,
} from './queueTraining';

const COURSE_490 = 'css-490-spring-2026-cgvl';

const PREPARED: CourseModelRequest = {
  courseId: COURSE_490,
  status: 'preparing',
  requestedAt: '2026-08-11T10:00:00.000Z',
  updatedAt: '2026-08-11T12:00:00.000Z',
  approvedExampleCount: 42,
  preparation: {
    preparedAt: '2026-08-11T12:00:00.000Z',
    sourceApprovedExampleCount: 44,
    datasetRef: `exports/${COURSE_490}`,
    trainExamples: 38,
    validationExamples: 4,
    splitSeed: 360,
  },
};

const QUEUED_RUN: TrainingRun = {
  runId: 'run-20260812t120000z-0a1b2c',
  courseId: COURSE_490,
  mode: 'full',
  state: 'queued',
  enqueuedAt: '2026-08-12T12:00:00.000Z',
  updatedAt: '2026-08-12T12:00:00.000Z',
  datasetRef: `exports/${COURSE_490}`,
  approvedExampleCount: 44,
  trainExamples: 38,
  validationExamples: 4,
  attempt: 0,
};

beforeEach(() => {
  vi.clearAllMocks();
  fetchCourseTrainingRuns.mockResolvedValue([]);
  enqueueTrainingRun.mockResolvedValue({
    courseId: COURSE_490,
    runId: QUEUED_RUN.runId,
    run: QUEUED_RUN,
  });
  updateCourseModelRequest.mockResolvedValue(undefined);
});

describe('canQueueTraining', () => {
  it('allows a prepared request with nothing outstanding', () => {
    expect(canQueueTraining(PREPARED, [])).toBe(true);
  });

  it('refuses anything unprepared', () => {
    expect(canQueueTraining(null)).toBe(false);
    expect(canQueueTraining({ ...PREPARED, preparation: undefined })).toBe(false);
    expect(canQueueTraining({ ...PREPARED, status: 'requested' })).toBe(false);
    expect(canQueueTraining({ ...PREPARED, status: 'ready' })).toBe(false);
  });

  it('refuses while a run is still outstanding', () => {
    expect(canQueueTraining(PREPARED, [QUEUED_RUN])).toBe(false);
    expect(canQueueTraining(PREPARED, [{ ...QUEUED_RUN, state: 'training' }])).toBe(
      false,
    );
  });

  it('allows another once the last run finished', () => {
    expect(canQueueTraining(PREPARED, [{ ...QUEUED_RUN, state: 'failed' }])).toBe(true);
  });
});

describe('queueTrainingForRequest', () => {
  it('enqueues the prepared dataset as it was recorded', async () => {
    const { run } = await queueTrainingForRequest(COURSE_490, PREPARED);

    expect(enqueueTrainingRun).toHaveBeenCalledWith(COURSE_490, {
      mode: 'full',
      datasetRef: `exports/${COURSE_490}`,
      approvedExampleCount: 44,
      trainExamples: 38,
      validationExamples: 4,
    });
    expect(run.state).toBe('queued');
  });

  it('queues a smoke run when asked for one', async () => {
    await queueTrainingForRequest(COURSE_490, PREPARED, 'smoke');
    expect(enqueueTrainingRun).toHaveBeenCalledWith(
      COURSE_490,
      expect.objectContaining({ mode: 'smoke' }),
    );
  });

  it('leaves the request at preparing and only points it at the run', async () => {
    await queueTrainingForRequest(COURSE_490, PREPARED);

    expect(updateCourseModelRequest).toHaveBeenCalledWith(COURSE_490, {
      currentRunId: QUEUED_RUN.runId,
      launchError: '',
    });
    // Nothing is training yet, so the professor-facing status must not say so.
    const patch = updateCourseModelRequest.mock.calls[0][1] as Record<string, unknown>;
    expect(patch.status).toBeUndefined();
  });

  it('refuses a request whose data was never prepared', async () => {
    await expect(
      queueTrainingForRequest(COURSE_490, { ...PREPARED, preparation: undefined }),
    ).rejects.toBeInstanceOf(TrainingNotPreparedError);
    expect(enqueueTrainingRun).not.toHaveBeenCalled();
  });

  it('refuses when a run is already outstanding', async () => {
    fetchCourseTrainingRuns.mockResolvedValue([QUEUED_RUN]);

    await expect(queueTrainingForRequest(COURSE_490, PREPARED)).rejects.toBeInstanceOf(
      DuplicateTrainingRunError,
    );
    expect(enqueueTrainingRun).not.toHaveBeenCalled();
    expect(updateCourseModelRequest).not.toHaveBeenCalled();
  });

  it('does not point the request at a run that was refused', async () => {
    enqueueTrainingRun.mockRejectedValue(new DuplicateTrainingRunError());

    await expect(queueTrainingForRequest(COURSE_490, PREPARED)).rejects.toBeInstanceOf(
      DuplicateTrainingRunError,
    );
    expect(updateCourseModelRequest).not.toHaveBeenCalled();
  });

  it('reads and writes only the course it was given', async () => {
    await queueTrainingForRequest(COURSE_490, PREPARED);

    expect(fetchCourseTrainingRuns).toHaveBeenCalledWith(COURSE_490);
    expect(enqueueTrainingRun.mock.calls[0][0]).toBe(COURSE_490);
    expect(updateCourseModelRequest.mock.calls[0][0]).toBe(COURSE_490);
  });

  it('rejects an invalid course id before any write', async () => {
    await expect(queueTrainingForRequest('../etc', PREPARED)).rejects.toThrow(
      /Invalid courseId/,
    );
    expect(fetchCourseTrainingRuns).not.toHaveBeenCalled();
  });
});

describe('the queue write goes through FastAPI to PostgreSQL', () => {
  it('calls the backend enqueue endpoint', async () => {
    await queueTrainingForRequest(COURSE_490, PREPARED);

    // The browser does not write the queue itself; the backend inserts the run
    // into PostgreSQL inside one transaction.
    expect(enqueueTrainingRun).toHaveBeenCalledTimes(1);
    expect(enqueueTrainingRun.mock.calls[0][0]).toBe(COURSE_490);
  });

  it('reports a duplicate the backend refused as the typed error', async () => {
    const { ApiError } = await import('./api');
    enqueueTrainingRun.mockRejectedValue(
      new ApiError('Course already has an active training run.', 409),
    );

    await expect(queueTrainingForRequest(COURSE_490, PREPARED)).rejects.toBeInstanceOf(
      DuplicateTrainingRunError,
    );
    expect(updateCourseModelRequest).not.toHaveBeenCalled();
  });

  it('lets a real backend failure through rather than calling it a duplicate', async () => {
    const { ApiError } = await import('./api');
    enqueueTrainingRun.mockRejectedValue(new ApiError('Service unavailable', 503));

    await expect(
      queueTrainingForRequest(COURSE_490, PREPARED),
    ).rejects.not.toBeInstanceOf(DuplicateTrainingRunError);
  });

  it('returns the queued run with no caveat about a second store', async () => {
    /*
     * There used to be a split-brain case here: the cluster had the run and
     * PostgreSQL — which the training list reads — did not, so the result
     * carried `mirroredToPostgres` and a warning to admit it. One store, one
     * transaction, so the state cannot exist and the fields are gone.
     */
    const result = await queueTrainingForRequest(COURSE_490, PREPARED);

    expect(result.run.runId).toBe(QUEUED_RUN.runId);
    expect(result).not.toHaveProperty('mirroredToPostgres');
    expect(result).not.toHaveProperty('warning');
  });

  it('points the model request at the run it just queued', async () => {
    await queueTrainingForRequest(COURSE_490, PREPARED);

    expect(updateCourseModelRequest).toHaveBeenCalledWith(COURSE_490, {
      currentRunId: QUEUED_RUN.runId,
      launchError: '',
    });
  });
});

/* -------------------------------------------------------------------------- *
 * Retry eligibility
 *
 * Advisory: it decides whether the control is offered. The backend decides,
 * under a row lock, whether the action happens. They are written to agree, and
 * the cases below are the same ones `test_training_retry.py` asserts on the
 * other side of the wire.
 * -------------------------------------------------------------------------- */

const STALE_RUN: TrainingRun = {
  runId: 'run-20260823t064333z-3c94f0',
  courseId: COURSE_490,
  mode: 'full',
  state: 'submitted',
  enqueuedAt: '2026-08-23T06:40:00.000Z',
  updatedAt: '2026-08-23T06:43:33.000Z',
  datasetRef: `exports/${COURSE_490}`,
  approvedExampleCount: 42,
  trainExamples: 37,
  validationExamples: 5,
  attempt: 1,
  jobId: '253552',
};

const STALE_REQUEST: CourseModelRequest = {
  ...PREPARED,
  status: 'training',
  currentRunId: STALE_RUN.runId,
};

describe('canRetryTraining', () => {
  const NOW = new Date('2026-08-26T17:00:00.000Z');

  it('offers a retry for a run stuck at submitted for days', () => {
    // Three days of silence. A healthy job's own callbacks would have moved
    // `updatedAt` long before this.
    expect(canRetryTraining(STALE_REQUEST, [STALE_RUN], NOW)).toBe(true);
  });

  it('offers a retry for a run stuck at training for days', () => {
    expect(
      canRetryTraining(STALE_REQUEST, [{ ...STALE_RUN, state: 'training' }], NOW),
    ).toBe(true);
  });

  it('refuses a run that reported minutes ago, whose job may still be alive', () => {
    /*
     * Neither side can see Slurm. Retiring a run whose job is still running
     * would leave two jobs writing the same adapter directory, and the backend
     * refuses this for the same reason — so offering the button would only
     * produce a 409.
     */
    expect(
      canRetryTraining(
        STALE_REQUEST,
        [{ ...STALE_RUN, updatedAt: '2026-08-26T16:50:00.000Z' }],
        NOW,
      ),
    ).toBe(false);
  });

  it('refuses a run still training as of an hour ago', () => {
    expect(
      canRetryTraining(
        STALE_REQUEST,
        [
          {
            ...STALE_RUN,
            state: 'training',
            updatedAt: '2026-08-26T16:00:00.000Z',
          },
        ],
        NOW,
      ),
    ).toBe(false);
  });

  it('agrees with the backend about where the six-hour boundary falls', () => {
    const at = (hoursAgo: number) =>
      new Date(NOW.getTime() - hoursAgo * 3_600_000).toISOString();

    expect(
      canRetryTraining(STALE_REQUEST, [{ ...STALE_RUN, updatedAt: at(5.9) }], NOW),
    ).toBe(false);
    expect(
      canRetryTraining(STALE_REQUEST, [{ ...STALE_RUN, updatedAt: at(6.1) }], NOW),
    ).toBe(true);
  });

  it('still offers a retry for a failed run of any age', () => {
    expect(
      canRetryTraining(
        STALE_REQUEST,
        [{ ...STALE_RUN, state: 'failed', updatedAt: NOW.toISOString() }],
        NOW,
      ),
    ).toBe(true);
  });

  it('offers a retry for a failed run', () => {
    expect(
      canRetryTraining(STALE_REQUEST, [{ ...STALE_RUN, state: 'failed' }], NOW),
    ).toBe(true);
  });

  it('refuses a healthy queued run, which is already what a retry would make', () => {
    expect(
      canRetryTraining(STALE_REQUEST, [{ ...STALE_RUN, state: 'queued' }], NOW),
    ).toBe(false);
  });

  it('refuses a run a worker still holds', () => {
    expect(
      canRetryTraining(
        STALE_REQUEST,
        [
          {
            ...STALE_RUN,
            state: 'claimed',
            claim: {
              owner: 'adam@tillicum',
              claimedAt: '2026-08-26T16:55:00.000Z',
              expiresAt: '2026-08-26T17:10:00.000Z',
            },
          },
        ],
        NOW,
      ),
    ).toBe(false);
  });

  it('offers a retry once the lease has expired', () => {
    expect(
      canRetryTraining(
        STALE_REQUEST,
        [
          {
            ...STALE_RUN,
            state: 'claimed',
            claim: {
              owner: 'adam@tillicum',
              claimedAt: '2026-08-26T15:00:00.000Z',
              expiresAt: '2026-08-26T15:15:00.000Z',
            },
          },
        ],
        NOW,
      ),
    ).toBe(true);
  });

  it('refuses a succeeded run', () => {
    expect(
      canRetryTraining(STALE_REQUEST, [{ ...STALE_RUN, state: 'succeeded' }], NOW),
    ).toBe(false);
  });

  it('refuses when there is no run to retry', () => {
    expect(canRetryTraining(STALE_REQUEST, [], NOW)).toBe(false);
    expect(canRetryTraining(null, [STALE_RUN], NOW)).toBe(false);
  });

  it('refuses when there is no prepared dataset to point a job at', () => {
    expect(
      canRetryTraining(
        { ...STALE_REQUEST, preparation: undefined },
        [{ ...STALE_RUN, datasetRef: '' }],
        NOW,
      ),
    ).toBe(false);
  });

  it('acts on the run the request points at, not merely the newest', () => {
    const newer: TrainingRun = { ...STALE_RUN, runId: 'run-later', state: 'queued' };
    expect(findRetryTargetRun(STALE_REQUEST, [STALE_RUN, newer])?.runId).toBe(
      STALE_RUN.runId,
    );
  });
});

describe('retryTrainingForRequest', () => {
  beforeEach(() => {
    retryTrainingRun.mockResolvedValue({
      courseId: COURSE_490,
      runId: 'run-20260826t170000z-9f0e1d',
      run: { ...STALE_RUN, runId: 'run-20260826t170000z-9f0e1d', state: 'queued' },
      supersededRunId: STALE_RUN.runId,
      supersededRun: { ...STALE_RUN, state: 'failed' },
      requestStatus: 'preparing',
    });
  });

  it('sends the course and nothing else', async () => {
    /*
     * Deliberately not "which run to retire" or "what to carry forward".
     * Anything the browser read is stale by the time the write lands; the
     * backend re-reads it all under a row lock.
     */
    await retryTrainingForRequest(COURSE_490);

    expect(retryTrainingRun).toHaveBeenCalledWith(COURSE_490);
  });

  it('returns the queued replacement and the run it retired', async () => {
    const result = await retryTrainingForRequest(COURSE_490);

    expect(result.run.runId).toBe('run-20260826t170000z-9f0e1d');
    expect(result.run.state).toBe('queued');
    expect(result.supersededRunId).toBe(STALE_RUN.runId);
  });

  it('rejects an invalid course id before any request is made', async () => {
    await expect(retryTrainingForRequest('../etc')).rejects.toThrow();
    expect(retryTrainingRun).not.toHaveBeenCalled();
  });

  it('lets a backend refusal reach the caller unchanged', async () => {
    retryTrainingRun.mockRejectedValue(
      new Error('This run is held by adam@tillicum until 2026-08-26T17:10:00+00:00.'),
    );

    await expect(retryTrainingForRequest(COURSE_490)).rejects.toThrow(
      /held by adam@tillicum/,
    );
  });
});
