import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./firebase', () => ({ database: {}, app: {} }));

const runTransactionMock = vi.fn();
const getMock = vi.fn();

vi.mock('firebase/database', () => ({
  ref: (_db: unknown, path: string) => ({ path }),
  get: (...args: unknown[]) => getMock(...args),
  onValue: vi.fn(),
  runTransaction: (...args: unknown[]) => runTransactionMock(...args),
}));

import {
  DuplicateTrainingRunError,
  enqueueTrainingRun,
  fetchCourseTrainingRuns,
  findActiveTrainingRun,
  generateTrainingRunId,
  isActiveTrainingRun,
  parseTrainingRun,
  parseTrainingRuns,
} from './trainingRunDb';
import { getCourseModelPath, getCourseTrainingRunsPath } from './coursePaths';

const COURSE_A = 'css-490-spring-2026-cgvl';
const COURSE_B = 'css-350-winter-2026-drlb';
const CSS_360 = 'css-360-winter-2026-a7rp';

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

const INPUT = {
  mode: 'full' as const,
  datasetRef: `exports/${COURSE_A}`,
  approvedExampleCount: 42,
  trainExamples: 38,
  validationExamples: 4,
};

/** What the enqueue transaction would write over `current`. */
function transactionResult(current: unknown) {
  const updater = runTransactionMock.mock.calls[0][1] as (value: unknown) => unknown;
  return updater(current);
}

beforeEach(() => {
  vi.clearAllMocks();
  runTransactionMock.mockImplementation(async (_ref, updater) => ({
    committed: updater(null) !== undefined,
    snapshot: null,
  }));
  getMock.mockResolvedValue({ exists: () => false, val: () => null });
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

describe('enqueueTrainingRun', () => {
  it('writes one queued run for that course', async () => {
    const run = await enqueueTrainingRun(COURSE_A, INPUT);

    expect(runTransactionMock).toHaveBeenCalledTimes(1);
    expect(runTransactionMock.mock.calls[0][0]).toEqual({
      path: getCourseTrainingRunsPath(COURSE_A),
    });
    expect(run.state).toBe('queued');
    expect(run.courseId).toBe(COURSE_A);
    expect(run.attempt).toBe(0);
    expect(run.trainExamples).toBe(38);
  });

  it('invents no job id and claims nothing', async () => {
    const run = await enqueueTrainingRun(COURSE_A, INPUT);
    const written = (transactionResult(null) as Record<string, unknown>)[run.runId];

    expect(run.claim).toBeUndefined();
    expect(Object.keys(written as object)).not.toContain('jobId');
    expect(JSON.stringify(written)).not.toMatch(/job/i);
  });

  it('refuses when this course already has an outstanding run', async () => {
    runTransactionMock.mockImplementation(async (_ref, updater) => ({
      committed: updater({ 'run-1': { ...STORED, state: 'claimed' } }) !== undefined,
      snapshot: null,
    }));

    await expect(enqueueTrainingRun(COURSE_A, INPUT)).rejects.toBeInstanceOf(
      DuplicateTrainingRunError,
    );
  });

  it('allows a new run once every earlier one finished', async () => {
    runTransactionMock.mockImplementation(async (_ref, updater) => ({
      committed:
        updater({
          'run-1': { ...STORED, state: 'succeeded' },
          'run-2': { ...STORED, state: 'failed' },
        }) !== undefined,
      snapshot: null,
    }));

    await expect(enqueueTrainingRun(COURSE_A, INPUT)).resolves.toBeTruthy();
  });

  it('keeps the runs that are already there', async () => {
    const run = await enqueueTrainingRun(COURSE_A, INPUT);
    const next = transactionResult({ 'run-old': { ...STORED, state: 'succeeded' } }) as
      | Record<string, unknown>
      | undefined;

    expect(Object.keys(next ?? {})).toEqual(['run-old', run.runId]);
  });

  it('rejects an unknown mode', async () => {
    await expect(
      enqueueTrainingRun(COURSE_A, { ...INPUT, mode: 'gigantic' as never }),
    ).rejects.toThrow(/Unknown training mode/);
  });

  it('rejects a course id that could escape its own path', async () => {
    for (const bad of ['', '../etc', 'CSS-360', 'a/b', 'x$y']) {
      await expect(enqueueTrainingRun(bad, INPUT)).rejects.toThrow(/Invalid courseId/);
    }
    expect(runTransactionMock).not.toHaveBeenCalled();
  });
});

describe('course isolation', () => {
  it('scopes every read and write to one course', async () => {
    await enqueueTrainingRun(COURSE_A, INPUT);
    await fetchCourseTrainingRuns(COURSE_B);

    expect(runTransactionMock.mock.calls[0][0]).toEqual({
      path: `courses/${COURSE_A}/trainingRuns`,
    });
    expect(getMock.mock.calls[0][0]).toEqual({
      path: `courses/${COURSE_B}/trainingRuns`,
    });
  });

  it("one course's outstanding run does not block another", async () => {
    // Course A is busy; the transaction for course B sees only B's own node.
    runTransactionMock.mockImplementation(async (ref, updater) => {
      const value =
        (ref as { path: string }).path === `courses/${COURSE_A}/trainingRuns`
          ? { 'run-1': { ...STORED, state: 'claimed' } }
          : null;
      return { committed: updater(value) !== undefined, snapshot: null };
    });

    await expect(enqueueTrainingRun(COURSE_A, INPUT)).rejects.toBeInstanceOf(
      DuplicateTrainingRunError,
    );
    await expect(
      enqueueTrainingRun(COURSE_B, { ...INPUT, datasetRef: `exports/${COURSE_B}` }),
    ).resolves.toBeTruthy();
  });

  it('never touches the CSS 360 model registry', async () => {
    await enqueueTrainingRun(CSS_360, { ...INPUT, datasetRef: `exports/${CSS_360}` });

    const paths = [
      ...runTransactionMock.mock.calls.map((call) => (call[0] as { path: string }).path),
      ...getMock.mock.calls.map((call) => (call[0] as { path: string }).path),
    ];
    expect(paths).toEqual([`courses/${CSS_360}/trainingRuns`]);
    expect(paths).not.toContain(getCourseModelPath(CSS_360));
  });
});
