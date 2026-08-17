import { beforeEach, describe, expect, it, vi } from 'vitest';

const listTrainingRunsMock = vi.fn();

// Reads come from PostgreSQL through FastAPI; only the queue write is Firebase.
vi.mock('./dbApi', () => ({
  listTrainingRuns: (...args: unknown[]) => listTrainingRunsMock(...args),
}));

import {
  fetchCourseTrainingRuns,
  findActiveTrainingRun,
  generateTrainingRunId,
  isActiveTrainingRun,
  parseTrainingRun,
  parseTrainingRuns,
} from './trainingRunDb';

const COURSE_A = 'css-490-spring-2026-cgvl';
const COURSE_B = 'css-350-winter-2026-drlb';

const STORED = {
  courseId: COURSE_A,
  mode: 'full',
  state: 'queued',
  enqueuedAt: '2026-08-12T10:00:00.000Z',
  updatedAt: '2026-08-12T10:00:00.000Z',
  datasetRef: `exports/${COURSE_A}`,
  approvedExampleCount: 42,
  trainExamples: 38,
  validationExamples: 4,
  attempt: 0,
};



beforeEach(() => {
  vi.clearAllMocks();
  listTrainingRunsMock.mockResolvedValue({ courseId: COURSE_A, count: 0, runs: [] });
});

describe('parseTrainingRun', () => {
  it('reads a stored run', () => {
    expect(parseTrainingRun('run-1', STORED)).toEqual({
      runId: 'run-1',
      courseId: COURSE_A,
      mode: 'full',
      state: 'queued',
      enqueuedAt: '2026-08-12T10:00:00.000Z',
      updatedAt: '2026-08-12T10:00:00.000Z',
      datasetRef: `exports/${COURSE_A}`,
      approvedExampleCount: 42,
      trainExamples: 38,
      validationExamples: 4,
      attempt: 0,
    });
  });

  it('keeps a claim that says who holds it and until when', () => {
    const run = parseTrainingRun('run-1', {
      ...STORED,
      state: 'claimed',
      attempt: 1,
      claim: {
        owner: 'alice@tillicum',
        claimedAt: '2026-08-12T11:00:00.000Z',
        expiresAt: '2026-08-12T11:15:00.000Z',
      },
    });
    expect(run?.claim?.owner).toBe('alice@tillicum');
    expect(run?.attempt).toBe(1);
  });

  it('keeps a real job id and ignores a blank one', () => {
    expect(parseTrainingRun('run-1', { ...STORED, jobId: '9182736' })?.jobId).toBe(
      '9182736',
    );
    expect(parseTrainingRun('run-1', { ...STORED, jobId: '' })?.jobId).toBeUndefined();
  });

  it('drops a claim that cannot be reasoned about', () => {
    const run = parseTrainingRun('run-1', {
      ...STORED,
      claim: { owner: '', claimedAt: '2026-08-12T11:00:00.000Z' },
    });
    expect(run?.claim).toBeUndefined();
  });

  it('rejects records that cannot be acted on', () => {
    expect(parseTrainingRun('run-1', null)).toBeNull();
    expect(parseTrainingRun('run-1', {})).toBeNull();
    expect(parseTrainingRun('run-1', { ...STORED, state: 'banana' })).toBeNull();
    expect(parseTrainingRun('run-1', { ...STORED, mode: 'gigantic' })).toBeNull();
    expect(parseTrainingRun('run-1', { ...STORED, courseId: '' })).toBeNull();
  });
});

describe('run states', () => {
  it('treats only succeeded and failed as finished', () => {
    const state = (value: string) =>
      isActiveTrainingRun(parseTrainingRun('run-1', { ...STORED, state: value })!);

    expect(state('queued')).toBe(true);
    expect(state('claimed')).toBe(true);
    expect(state('submitted')).toBe(true);
    expect(state('training')).toBe(true);
    expect(state('succeeded')).toBe(false);
    expect(state('failed')).toBe(false);
  });

  it('finds the outstanding run among finished ones', () => {
    const runs = parseTrainingRuns({
      'run-old': { ...STORED, state: 'failed', enqueuedAt: '2026-08-01T10:00:00.000Z' },
      'run-now': { ...STORED, state: 'submitted' },
    });
    expect(findActiveTrainingRun(runs)?.runId).toBe('run-now');
  });

  it('has nothing outstanding when every run finished', () => {
    const runs = parseTrainingRuns({
      'run-a': { ...STORED, state: 'succeeded' },
      'run-b': { ...STORED, state: 'failed' },
    });
    expect(findActiveTrainingRun(runs)).toBeNull();
  });
});

describe('generateTrainingRunId', () => {
  it('produces a legal, sortable key', () => {
    const early = generateTrainingRunId(new Date('2026-08-12T10:00:00.000Z'));
    const late = generateTrainingRunId(new Date('2026-08-12T11:00:00.000Z'));

    expect(early).toMatch(/^[a-z0-9]+(?:-[a-z0-9]+)*$/);
    // Firebase keys cannot contain any of these.
    expect(early).not.toMatch(/[.$#[\]/]/);
    expect(early < late).toBe(true);
  });
});

describe('course isolation', () => {
  it('reads only the course it was given', async () => {
    await fetchCourseTrainingRuns(COURSE_B);

    expect(listTrainingRunsMock).toHaveBeenCalledWith(COURSE_B);
    expect(listTrainingRunsMock).not.toHaveBeenCalledWith(COURSE_A);
  });

  it('rejects a course id that could escape its own path', async () => {
    for (const bad of ['../css-360', 'CSS 360', 'css_360', '']) {
      await expect(fetchCourseTrainingRuns(bad)).rejects.toThrow(/Invalid courseId/);
    }
    expect(listTrainingRunsMock).not.toHaveBeenCalled();
  });

  it('shows a run the backend queued, keyed by its run id', async () => {
    // Proof that a newly enqueued run is visible through the PostgreSQL read
    // path: the queue write happens in FastAPI, and this is what the admin
    // page sees afterwards.
    listTrainingRunsMock.mockResolvedValue({
      courseId: COURSE_A,
      count: 1,
      runs: [{ ...STORED, runId: 'run-20260812t100000z-abc123' }],
    });

    const runs = await fetchCourseTrainingRuns(COURSE_A);

    expect(runs).toHaveLength(1);
    expect(runs[0].runId).toBe('run-20260812t100000z-abc123');
    expect(runs[0].state).toBe('queued');
  });
});
