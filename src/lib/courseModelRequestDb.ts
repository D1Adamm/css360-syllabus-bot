import {
  get,
  onValue,
  ref,
  runTransaction,
  update,
  type Unsubscribe,
} from 'firebase/database';
import type {
  CourseModelRequest,
  CourseModelRequestPreparation,
  CourseModelRequestStatus,
  CourseModelRequestTraining,
} from '../types';
import { assertValidCourseId } from './courseId';
import { getCourseModelRequestPath } from './coursePaths';
import { database } from './firebase';

/**
 * Course model requests at `courses/{courseId}/modelRequest`.
 *
 * Course-scoped like everything else under `courses/`: the path is built from a
 * validated id, so one course can never read or write another's request.
 *
 * Kept separate from the model registry on purpose. The registry is a record of
 * artifacts that exist; a request is work that has been asked for. Writing a
 * pending request into the registry would make it look like a course had a
 * model when it does not.
 */

const REQUEST_STATUSES: readonly CourseModelRequestStatus[] = [
  'requested',
  'preparing',
  'training',
  'ready',
  'failed',
];

/**
 * Statuses that mean work is outstanding.
 *
 * Only these block a second request. `ready` and `failed` are terminal — a
 * failed request should not lock a course out forever.
 */
const ACTIVE_STATUSES: readonly CourseModelRequestStatus[] = [
  'requested',
  'preparing',
  'training',
];

export function isRequestStatus(value: unknown): value is CourseModelRequestStatus {
  return (
    typeof value === 'string' &&
    REQUEST_STATUSES.includes(value as CourseModelRequestStatus)
  );
}

export function isActiveRequest(request: CourseModelRequest | null): boolean {
  return request !== null && ACTIVE_STATUSES.includes(request.status);
}

export function parseCourseModelRequest(value: unknown): CourseModelRequest | null {
  if (!value || typeof value !== 'object') {
    return null;
  }

  const record = value as Record<string, unknown>;

  if (
    typeof record.courseId !== 'string' ||
    record.courseId.trim() === '' ||
    typeof record.requestedAt !== 'string' ||
    !isRequestStatus(record.status)
  ) {
    return null;
  }

  const count = Number(record.approvedExampleCount);
  const preparation = parsePreparation(record.preparation);
  const training = parseTraining(record.training);

  return {
    courseId: record.courseId,
    status: record.status,
    requestedAt: record.requestedAt,
    updatedAt:
      typeof record.updatedAt === 'string' ? record.updatedAt : record.requestedAt,
    approvedExampleCount: Number.isFinite(count) && count >= 0 ? count : 0,
    ...(typeof record.failureMessage === 'string'
      ? { failureMessage: record.failureMessage }
      : {}),
    ...(preparation ? { preparation } : {}),
    ...(typeof record.preparationError === 'string'
      ? { preparationError: record.preparationError }
      : {}),
    ...(training ? { training } : {}),
    ...(typeof record.launchError === 'string'
      ? { launchError: record.launchError }
      : {}),
  };
}

function parseTraining(value: unknown): CourseModelRequestTraining | null {
  if (!value || typeof value !== 'object') {
    return null;
  }

  const record = value as Record<string, unknown>;

  // A training block without a real job id is meaningless — it would claim a
  // job exists that nothing can look up.
  if (
    typeof record.jobId !== 'string' ||
    record.jobId.trim() === '' ||
    typeof record.submittedAt !== 'string'
  ) {
    return null;
  }

  const asCount = (raw: unknown) => {
    const value = Number(raw);
    return Number.isFinite(value) && value >= 0 ? value : 0;
  };

  return {
    jobId: record.jobId,
    mode: typeof record.mode === 'string' ? record.mode : 'full',
    submittedAt: record.submittedAt,
    datasetRef: typeof record.datasetRef === 'string' ? record.datasetRef : '',
    trainExamples: asCount(record.trainExamples),
    validationExamples: asCount(record.validationExamples),
  };
}

function parsePreparation(value: unknown): CourseModelRequestPreparation | null {
  if (!value || typeof value !== 'object') {
    return null;
  }

  const record = value as Record<string, unknown>;

  if (
    typeof record.preparedAt !== 'string' ||
    typeof record.datasetRef !== 'string' ||
    record.datasetRef.trim() === ''
  ) {
    return null;
  }

  const asCount = (raw: unknown) => {
    const value = Number(raw);
    return Number.isFinite(value) && value >= 0 ? value : 0;
  };

  return {
    preparedAt: record.preparedAt,
    sourceApprovedExampleCount: asCount(record.sourceApprovedExampleCount),
    datasetRef: record.datasetRef,
    trainExamples: asCount(record.trainExamples),
    validationExamples: asCount(record.validationExamples),
    ...(Number.isFinite(Number(record.splitSeed))
      ? { splitSeed: Number(record.splitSeed) }
      : {}),
  };
}

export function getCourseModelRequestRef(courseId: string) {
  assertValidCourseId(courseId);
  return ref(database, getCourseModelRequestPath(courseId));
}

export async function fetchCourseModelRequest(
  courseId: string,
): Promise<CourseModelRequest | null> {
  const snapshot = await get(getCourseModelRequestRef(courseId));
  return snapshot.exists() ? parseCourseModelRequest(snapshot.val()) : null;
}

export function subscribeToCourseModelRequest(
  courseId: string,
  onData: (request: CourseModelRequest | null) => void,
  onError?: (message: string) => void,
): Unsubscribe {
  assertValidCourseId(courseId);

  return onValue(
    getCourseModelRequestRef(courseId),
    (snapshot) => {
      onData(snapshot.exists() ? parseCourseModelRequest(snapshot.val()) : null);
    },
    (error) => {
      onError?.(error.message);
    },
  );
}

export class DuplicateModelRequestError extends Error {
  constructor() {
    super('A model request is already in progress for this course.');
    this.name = 'DuplicateModelRequestError';
  }
}

/**
 * Creates a request, refusing if one is already outstanding.
 *
 * Uses a transaction rather than read-then-write: a professor double-clicking,
 * or two browser tabs open on the same course, would otherwise both read "no
 * request" and both write one. The transaction aborts when the current value is
 * already an active request, so exactly one write wins.
 */
export async function createCourseModelRequest(
  courseId: string,
  approvedExampleCount: number,
): Promise<CourseModelRequest> {
  assertValidCourseId(courseId);

  const now = new Date().toISOString();
  const request: CourseModelRequest = {
    courseId,
    status: 'requested',
    requestedAt: now,
    updatedAt: now,
    approvedExampleCount: Math.max(0, Math.trunc(approvedExampleCount)),
  };

  const result = await runTransaction(
    getCourseModelRequestRef(courseId),
    (current: unknown) => {
      const existing = parseCourseModelRequest(current);
      if (isActiveRequest(existing)) {
        // Abort: leave the existing request exactly as it is.
        return undefined;
      }
      return request;
    },
  );

  if (!result.committed) {
    throw new DuplicateModelRequestError();
  }

  return request;
}

/**
 * Patches a course's request in place.
 *
 * A merge rather than a write: preparation records extra fields onto a request
 * that already exists, and must not clobber `requestedAt` or the professor's
 * original approved count.
 */
export async function updateCourseModelRequest(
  courseId: string,
  patch: Partial<CourseModelRequest>,
): Promise<void> {
  assertValidCourseId(courseId);
  await update(getCourseModelRequestRef(courseId), {
    ...patch,
    updatedAt: new Date().toISOString(),
  });
}
