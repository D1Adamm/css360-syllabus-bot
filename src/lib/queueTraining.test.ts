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
    enqueueTrainingRun: (...args: unknown[]) => enqueueTrainingRun(...args),
    fetchCourseTrainingRuns: (...args: unknown[]) => fetchCourseTrainingRuns(...args),
  };
});

vi.mock('./courseModelRequestDb', () => ({
  updateCourseModelRequest: (...args: unknown[]) => updateCourseModelRequest(...args),
}));

// The data layer imports Firebase through `./firebase`; the real module needs
// browser env vars that a unit test has no business providing.
vi.mock('./firebase', () => ({ database: {}, app: {} }));
vi.mock('firebase/database', () => ({
  ref: (_db: unknown, path: string) => ({ path }),
  get: vi.fn(),
  onValue: vi.fn(),
  runTransaction: vi.fn(),
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
  enqueueTrainingRun.mockResolvedValue(QUEUED_RUN);
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
