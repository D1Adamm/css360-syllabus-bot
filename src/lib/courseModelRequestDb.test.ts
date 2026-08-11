import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./firebase', () => ({ database: {}, app: {} }));

const runTransactionMock = vi.fn();

vi.mock('firebase/database', () => ({
  ref: (_db: unknown, path: string) => ({ path }),
  get: vi.fn(),
  onValue: vi.fn(),
  runTransaction: (...args: unknown[]) => runTransactionMock(...args),
}));

import {
  createCourseModelRequest,
  DuplicateModelRequestError,
  isActiveRequest,
  parseCourseModelRequest,
} from './courseModelRequestDb';
import { getCourseModelRequestPath } from './coursePaths';

const COURSE_A = 'css-490-spring-2026-cgvl';
const COURSE_B = 'css-350-winter-2026-drlb';

const STORED = {
  courseId: COURSE_A,
  status: 'requested',
  requestedAt: '2026-08-11T10:00:00.000Z',
  updatedAt: '2026-08-11T10:00:00.000Z',
  approvedExampleCount: 42,
};

/** Runs the transaction updater against `current` and reports the outcome. */
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
});

describe('parseCourseModelRequest', () => {
  it('reads a stored request', () => {
    expect(parseCourseModelRequest(STORED)).toEqual({
      courseId: COURSE_A,
      status: 'requested',
      requestedAt: '2026-08-11T10:00:00.000Z',
      updatedAt: '2026-08-11T10:00:00.000Z',
      approvedExampleCount: 42,
    });
  });

  it('keeps a recorded failure message', () => {
    const parsed = parseCourseModelRequest({
      ...STORED,
      status: 'failed',
      failureMessage: 'run exited non-zero',
    });
    expect(parsed?.failureMessage).toBe('run exited non-zero');
  });

  it('rejects records that are not usable', () => {
    expect(parseCourseModelRequest(null)).toBeNull();
    expect(parseCourseModelRequest({})).toBeNull();
    expect(parseCourseModelRequest({ ...STORED, status: 'queued' })).toBeNull();
    expect(parseCourseModelRequest({ ...STORED, courseId: '' })).toBeNull();
  });

  it('falls back to requestedAt when updatedAt is missing', () => {
    const { updatedAt, ...withoutUpdated } = STORED;
    void updatedAt;
    expect(parseCourseModelRequest(withoutUpdated)?.updatedAt).toBe(STORED.requestedAt);
  });
});

describe('isActiveRequest', () => {
  it('treats only outstanding work as active', () => {
    for (const status of ['requested', 'preparing', 'training'] as const) {
      expect(isActiveRequest({ ...STORED, status })).toBe(true);
    }
    // Terminal states must not lock a course out of ever requesting again.
    for (const status of ['ready', 'failed'] as const) {
      expect(isActiveRequest({ ...STORED, status })).toBe(false);
    }
    expect(isActiveRequest(null)).toBe(false);
  });
});

describe('createCourseModelRequest', () => {
  it('writes to the requesting course’s own path', async () => {
    await createCourseModelRequest(COURSE_A, 42);

    expect(runTransactionMock.mock.calls[0][0]).toEqual({
      path: getCourseModelRequestPath(COURSE_A),
    });
    expect(getCourseModelRequestPath(COURSE_A)).toBe(`courses/${COURSE_A}/modelRequest`);
    expect(getCourseModelRequestPath(COURSE_B)).not.toBe(
      getCourseModelRequestPath(COURSE_A),
    );
  });

  it('creates a requested record carrying the approved count', async () => {
    const request = await createCourseModelRequest(COURSE_A, 42);

    expect(request.courseId).toBe(COURSE_A);
    expect(request.status).toBe('requested');
    expect(request.approvedExampleCount).toBe(42);
    expect(request.requestedAt).toBe(request.updatedAt);
  });

  it('writes when nothing exists yet', () => {
    void createCourseModelRequest(COURSE_A, 42);
    expect(transactionResult(null)).toMatchObject({ status: 'requested' });
  });

  it('aborts rather than overwriting an outstanding request', async () => {
    void createCourseModelRequest(COURSE_A, 42);

    // A second attempt while work is outstanding must leave the record alone.
    for (const status of ['requested', 'preparing', 'training'] as const) {
      expect(transactionResult({ ...STORED, status })).toBeUndefined();
    }
  });

  it('allows a new request after a terminal one', () => {
    void createCourseModelRequest(COURSE_A, 42);

    for (const status of ['ready', 'failed'] as const) {
      expect(transactionResult({ ...STORED, status })).toMatchObject({
        status: 'requested',
      });
    }
  });

  it('surfaces a duplicate as a typed error when the transaction aborts', async () => {
    runTransactionMock.mockResolvedValue({ committed: false, snapshot: null });

    await expect(createCourseModelRequest(COURSE_A, 42)).rejects.toBeInstanceOf(
      DuplicateModelRequestError,
    );
  });

  it('refuses an unsafe course id before touching the database', async () => {
    await expect(createCourseModelRequest('Bad_Id', 1)).rejects.toThrow();
    expect(runTransactionMock).not.toHaveBeenCalled();
  });

  it('normalises a nonsense approved count', async () => {
    expect((await createCourseModelRequest(COURSE_A, -5)).approvedExampleCount).toBe(0);
    expect((await createCourseModelRequest(COURSE_A, 4.7)).approvedExampleCount).toBe(4);
  });
});
