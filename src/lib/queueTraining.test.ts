import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { CourseModelRequest, TrainingRun } from '../types';

const enqueueTrainingRun = vi.fn();
const fetchCourseTrainingRuns = vi.fn();
const updateCourseModelRequest = vi.fn();

vi.mock('./trainingRunDb', async () => {
  const actual = await vi.importActual<typeof import('./trainingRunDb')>(
    './trainingRunDb',
  );
  return {
    ...actual,
    fetchCourseTrainingRuns: (...args: unknown[]) => fetchCourseTrainingRuns(...args),
  };
});

// Queueing now goes through the backend, which writes the Firebase queue the
// cluster claims from and mirrors the run into PostgreSQL.
vi.mock('./api', async () => {
  const actual = await vi.importActual<typeof import('./api')>('./api');
  return {
    ...actual,
    enqueueTrainingRun: (...args: unknown[]) => enqueueTrainingRun(...args),
  };
});

vi.mock('./courseModelRequestDb', () => ({
  updateCourseModelRequest: (...args: unknown[]) => updateCourseModelRequest(...args),
}));

import { DuplicateTrainingRunError } from './trainingRunDb';
import {
  canQueueTraining,
  queueTrainingForRequest,
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
    mirroredToPostgres: true,
    warning: null,
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

describe('the queue write goes through FastAPI, never Firebase', () => {
  it('calls the backend enqueue endpoint', async () => {
    await queueTrainingForRequest(COURSE_490, PREPARED);

    // The browser no longer writes the Firebase queue itself; the backend does,
    // and it mirrors the run into PostgreSQL in the same request.
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

  it('passes through a known-stale PostgreSQL mirror instead of implying success', async () => {
    /*
     * The split-brain case. The cluster has the run; PostgreSQL — which the
     * training list reads — does not. The caller has to be able to say that,
     * so the result carries it rather than looking like an ordinary success.
     */
    enqueueTrainingRun.mockResolvedValue({
      courseId: COURSE_490,
      runId: QUEUED_RUN.runId,
      run: QUEUED_RUN,
      mirroredToPostgres: false,
      warning: 'Queued for the cluster but not recorded in PostgreSQL.',
    });

    const result = await queueTrainingForRequest(COURSE_490, PREPARED);

    expect(result.mirroredToPostgres).toBe(false);
    expect(result.warning).toContain('not recorded in PostgreSQL');
  });

  it('reports the mirror as healthy on a normal queue', async () => {
    const result = await queueTrainingForRequest(COURSE_490, PREPARED);

    expect(result.mirroredToPostgres).toBe(true);
    expect(result.warning).toBeNull();
  });
});
