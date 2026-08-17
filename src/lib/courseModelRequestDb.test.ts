import { beforeEach, describe, expect, it, vi } from 'vitest';

const createModelRequestMock = vi.fn();
const getModelRequestMock = vi.fn();
const updateModelRequestMock = vi.fn();

// Model requests are persisted through FastAPI into PostgreSQL. The
// one-active-request guard moved into the database with the record, so the
// duplicate case arrives as a 409 rather than an aborted transaction.
vi.mock('./dbApi', () => ({
  createModelRequest: (...args: unknown[]) => createModelRequestMock(...args),
  getModelRequest: (...args: unknown[]) => getModelRequestMock(...args),
  updateModelRequest: (...args: unknown[]) => updateModelRequestMock(...args),
}));

import {
  createCourseModelRequest,
  DuplicateModelRequestError,
  fetchCourseModelRequest,
  isActiveRequest,
  parseCourseModelRequest,
} from './courseModelRequestDb';
import { ApiError } from './api';

const COURSE_A = 'css-490-spring-2026-cgvl';
const COURSE_B = 'css-350-winter-2026-drlb';

const STORED = {
  courseId: COURSE_A,
  status: 'requested',
  requestedAt: '2026-08-11T10:00:00.000Z',
  updatedAt: '2026-08-11T10:00:00.000Z',
  approvedExampleCount: 42,
};

beforeEach(() => {
  vi.clearAllMocks();
  // The backend echoes back the record it stored.
  createModelRequestMock.mockImplementation(
    async (courseId: string, approvedExampleCount: number) => ({
      ...STORED,
      courseId,
      approvedExampleCount,
    }),
  );
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
  it('requests through the API for that course only', async () => {
    await createCourseModelRequest(COURSE_A, 42);

    expect(createModelRequestMock).toHaveBeenCalledWith(COURSE_A, 42);
    expect(createModelRequestMock).not.toHaveBeenCalledWith(
      COURSE_B,
      expect.anything(),
    );
  });

  it('creates a requested record carrying the approved count', async () => {
    const request = await createCourseModelRequest(COURSE_A, 42);

    expect(request.courseId).toBe(COURSE_A);
    expect(request.status).toBe('requested');
    expect(request.approvedExampleCount).toBe(42);
  });

  it('surfaces a duplicate as a typed error when the backend refuses', async () => {
    // The guard is now a conditional insert; a refused create is a 409.
    createModelRequestMock.mockRejectedValue(
      new ApiError('Course already has an outstanding model request.', 409),
    );

    await expect(createCourseModelRequest(COURSE_A, 42)).rejects.toBeInstanceOf(
      DuplicateModelRequestError,
    );
  });

  it('lets a real failure through rather than calling it a duplicate', async () => {
    createModelRequestMock.mockRejectedValue(new ApiError('Service unavailable', 503));

    await expect(createCourseModelRequest(COURSE_A, 42)).rejects.not.toBeInstanceOf(
      DuplicateModelRequestError,
    );
  });

  it('refuses an unsafe course id before touching the backend', async () => {
    await expect(createCourseModelRequest('Bad_Id', 1)).rejects.toThrow();
    expect(createModelRequestMock).not.toHaveBeenCalled();
  });

  it('normalises a nonsense approved count', async () => {
    await createCourseModelRequest(COURSE_A, -5);
    expect(createModelRequestMock).toHaveBeenLastCalledWith(COURSE_A, 0);

    await createCourseModelRequest(COURSE_A, 4.7);
    expect(createModelRequestMock).toHaveBeenLastCalledWith(COURSE_A, 4);
  });
});

describe('fetchCourseModelRequest', () => {
  it('reads the request for that course', async () => {
    getModelRequestMock.mockResolvedValue(STORED);

    const request = await fetchCourseModelRequest(COURSE_A);

    expect(getModelRequestMock).toHaveBeenCalledWith(COURSE_A);
    expect(request?.status).toBe('requested');
  });

  it('reports a course that never requested one as null, not an error', async () => {
    getModelRequestMock.mockRejectedValue(new ApiError('not found', 404));

    await expect(fetchCourseModelRequest(COURSE_A)).resolves.toBeNull();
  });

  it('lets a read failure surface rather than claiming nothing was requested', async () => {
    getModelRequestMock.mockRejectedValue(new ApiError('Service unavailable', 503));

    await expect(fetchCourseModelRequest(COURSE_A)).rejects.toThrow();
  });
});
