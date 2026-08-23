import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { CourseModelRequest } from '../types';

const launchCourseTraining = vi.fn();
const updateCourseModelRequest = vi.fn();

vi.mock('./adminApi', () => ({
  launchCourseTraining: (...args: unknown[]) => launchCourseTraining(...args),
}));

vi.mock('./courseModelRequestDb', () => ({
  updateCourseModelRequest: (...args: unknown[]) => updateCourseModelRequest(...args),
}));

import {
  AlreadyLaunchedError,
  canLaunchTraining,
  launchTrainingForRequest,
  NotPreparedError,
} from './launchTraining';

const COURSE_490 = 'css-490-spring-2026-cgvl';
const COURSE_360 = 'css-360-winter-2026-a7rp';

const PREPARED: CourseModelRequest = {
  courseId: COURSE_490,
  status: 'preparing',
  requestedAt: '2026-08-11T10:00:00.000Z',
  updatedAt: '2026-08-11T12:00:00.000Z',
  approvedExampleCount: 42,
  preparation: {
    preparedAt: '2026-08-11T12:00:00.000Z',
    sourceApprovedExampleCount: 42,
    datasetRef: `exports/${COURSE_490}`,
    trainExamples: 38,
    validationExamples: 4,
    splitSeed: 360,
  },
};

beforeEach(() => {
  vi.clearAllMocks();
  launchCourseTraining.mockResolvedValue({
    courseId: COURSE_490,
    jobId: '9182736',
    mode: 'full',
    submittedAt: '2026-08-11T13:00:00.000Z',
    trainCount: 38,
    validationCount: 4,
  });
  updateCourseModelRequest.mockResolvedValue(undefined);
});

describe('canLaunchTraining', () => {
  it('allows only a prepared request with no job yet', () => {
    expect(canLaunchTraining(PREPARED)).toBe(true);
  });

  it('refuses anything unprepared or already launched', () => {
    expect(canLaunchTraining(null)).toBe(false);
    // Requested but never prepared.
    expect(
      canLaunchTraining({ ...PREPARED, status: 'requested', preparation: undefined }),
    ).toBe(false);
    // Preparing but preparation metadata missing.
    expect(canLaunchTraining({ ...PREPARED, preparation: undefined })).toBe(false);
    // Already has a job.
    expect(
      canLaunchTraining({
        ...PREPARED,
        training: {
          jobId: '1',
          mode: 'full',
          submittedAt: 'x',
          datasetRef: 'y',
          trainExamples: 1,
          validationExamples: 1,
        },
      }),
    ).toBe(false);
    // Terminal states.
    expect(canLaunchTraining({ ...PREPARED, status: 'ready' })).toBe(false);
    expect(canLaunchTraining({ ...PREPARED, status: 'failed' })).toBe(false);
  });
});

describe('launch preconditions', () => {
  it('refuses to launch a request that was never prepared', async () => {
    await expect(
      launchTrainingForRequest(COURSE_490, {
        ...PREPARED,
        status: 'requested',
        preparation: undefined,
      }),
    ).rejects.toBeInstanceOf(NotPreparedError);

    // The infrastructure boundary is never reached.
    expect(launchCourseTraining).not.toHaveBeenCalled();
  });

  it('refuses a second launch for the same course', async () => {
    await expect(
      launchTrainingForRequest(COURSE_490, {
        ...PREPARED,
        training: {
          jobId: '9182736',
          mode: 'full',
          submittedAt: '2026-08-11T13:00:00.000Z',
          datasetRef: `exports/${COURSE_490}`,
          trainExamples: 38,
          validationExamples: 4,
        },
      }),
    ).rejects.toBeInstanceOf(AlreadyLaunchedError);

    expect(launchCourseTraining).not.toHaveBeenCalled();
    // A duplicate attempt must not rewrite the request either.
    expect(updateCourseModelRequest).not.toHaveBeenCalled();
  });

  it('refuses an unsafe course id before anything else', async () => {
    await expect(launchTrainingForRequest('Bad_Id', PREPARED)).rejects.toThrow();
    expect(launchCourseTraining).not.toHaveBeenCalled();
  });
});

describe('successful launch', () => {
  it('persists the real Slurm job id', async () => {
    const { training } = await launchTrainingForRequest(COURSE_490, PREPARED);

    expect(training.jobId).toBe('9182736');
    expect(training.trainExamples).toBe(38);
    expect(training.validationExamples).toBe(4);
  });

  it('marks the request training only after submission returns a job id', async () => {
    let statusWhenCalled: string | undefined;
    launchCourseTraining.mockImplementation(async () => {
      // Nothing may have been written before the job exists.
      statusWhenCalled = updateCourseModelRequest.mock.calls[0]?.[1]?.status;
      return {
        courseId: COURSE_490,
        jobId: '9182736',
        mode: 'full',
        submittedAt: '2026-08-11T13:00:00.000Z',
        trainCount: 38,
        validationCount: 4,
      };
    });

    await launchTrainingForRequest(COURSE_490, PREPARED);

    expect(statusWhenCalled).toBeUndefined();
    const [courseId, patch] = updateCourseModelRequest.mock.calls[0];
    expect(courseId).toBe(COURSE_490);
    expect(patch.status).toBe('training');
    expect(patch.training.jobId).toBe('9182736');
  });

  it('keeps the relative dataset reference rather than anything absolute', async () => {
    const { training } = await launchTrainingForRequest(COURSE_490, PREPARED);

    expect(training.datasetRef).toBe(`exports/${COURSE_490}`);
    expect(training.datasetRef.startsWith('/')).toBe(false);
  });

  it('clears a previous launch error', async () => {
    await launchTrainingForRequest(COURSE_490, {
      ...PREPARED,
      launchError: 'rsync failed',
    });
    expect(updateCourseModelRequest.mock.calls[0][1].launchError).toBe('');
  });
});

describe('failure handling', () => {
  it('leaves the request at preparing when the sync fails', async () => {
    launchCourseTraining.mockRejectedValue(
      new Error('Syncing training data failed: connection closed'),
    );

    await expect(launchTrainingForRequest(COURSE_490, PREPARED)).rejects.toThrow(
      /Syncing training data failed/,
    );

    const [, patch] = updateCourseModelRequest.mock.calls[0];
    // Never `training` — nothing was submitted.
    expect(patch.status).toBe('preparing');
    expect(patch.launchError).toMatch(/Syncing training data failed/);
    expect(patch.training).toBeUndefined();
  });

  it('leaves the request at preparing when submission fails', async () => {
    launchCourseTraining.mockRejectedValue(
      new Error('Submitting the training job failed: Invalid account'),
    );

    await expect(launchTrainingForRequest(COURSE_490, PREPARED)).rejects.toThrow();
    expect(updateCourseModelRequest.mock.calls[0][1].status).toBe('preparing');
  });

  it('does not claim success when no job id came back', async () => {
    launchCourseTraining.mockRejectedValue(
      new Error('The training job did not report a Slurm job ID; nothing was submitted.'),
    );

    await expect(launchTrainingForRequest(COURSE_490, PREPARED)).rejects.toThrow();
    expect(updateCourseModelRequest.mock.calls[0][1].status).not.toBe('training');
  });

  it('is retryable after a transient failure', async () => {
    launchCourseTraining.mockRejectedValueOnce(new Error('transient'));

    await expect(launchTrainingForRequest(COURSE_490, PREPARED)).rejects.toThrow();
    await expect(launchTrainingForRequest(COURSE_490, PREPARED)).resolves.toMatchObject({
      training: { jobId: '9182736' },
    });

    expect(updateCourseModelRequest.mock.calls.at(-1)?.[1].status).toBe('training');
  });

  it('still reports the original failure when recording it also fails', async () => {
    launchCourseTraining.mockRejectedValue(new Error('submission blew up'));
    updateCourseModelRequest.mockRejectedValue(new Error('the database is down'));

    await expect(launchTrainingForRequest(COURSE_490, PREPARED)).rejects.toThrow(
      'submission blew up',
    );
  });
});

describe('course isolation', () => {
  it('launches and records against one course only', async () => {
    await launchTrainingForRequest(COURSE_490, PREPARED);

    for (const mock of [launchCourseTraining, updateCourseModelRequest]) {
      for (const call of mock.mock.calls) {
        expect(call[0]).toBe(COURSE_490);
        expect(call[0]).not.toBe(COURSE_360);
      }
    }
  });

  it('never carries another course’s dataset reference', async () => {
    const { training } = await launchTrainingForRequest(COURSE_490, PREPARED);
    expect(training.datasetRef).not.toContain(COURSE_360);
    expect(training.datasetRef).toContain(COURSE_490);
  });
});

describe('scope', () => {
  it('runs no infrastructure command from the browser', async () => {
    const fs = await import('node:fs');
    const raw = fs.readFileSync('src/lib/launchTraining.ts', 'utf8');

    // Strip comments: the docblock legitimately names what this module avoids.
    const code = raw
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .replace(/^\s*\/\/.*$/gm, '');

    // Everything infrastructural is behind the backend endpoint.
    expect(code).not.toMatch(/child_process|exec\(|spawn\(|rsync|scp\b/);
    expect(code).not.toMatch(/hyak|gpfs|tillicum|huggingface|sbatch|ssh/i);
    // And nothing here promotes or registers a model.
    expect(code).not.toMatch(/courseModelDb|promote|registerCourseModel/);
  });
});
