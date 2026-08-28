import * as dbApi from './dbApi';
import { pollingSubscription, type Unsubscribe } from './pollingSubscription';
import type {
  CourseModelRequest,
  CourseModelRequestPreparation,
  CourseModelRequestStatus,
  CourseModelRequestTraining,
} from '../types';
import { assertValidCourseId } from './courseId';

/**
 * Course model requests: the `model_requests` table, read through `/api/db`.
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
    // A pointer to the operational record, never the record itself. Everything
    // about the run lives in `training_runs`.
    ...(typeof record.currentRunId === 'string' && record.currentRunId.trim() !== ''
      ? { currentRunId: record.currentRunId }
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

/**
 * The model request for one course, or null when there is none.
 *
 * Read from PostgreSQL through FastAPI. A 404 is "never requested", which the
 * professor page renders as an offer to request one — quite different from a
 * read failure, which must not put that button in front of someone whose
 * request is already running.
 */
export async function fetchCourseModelRequest(
  courseId: string,
): Promise<CourseModelRequest | null> {
  assertValidCourseId(courseId);

  try {
    return parseCourseModelRequest(await dbApi.getModelRequest(courseId));
  } catch (error) {
    if (error instanceof Error && 'status' in error && error.status === 404) {
      return null;
    }
    throw error;
  }
}

/**
 * Watches a request while it is still outstanding.
 *
 * Preparation and training are driven by an administrator and a cluster, so a
 * professor's page has no other way to learn that its state moved. Polling
 * stops at `ready` and `failed` — the two states nothing follows.
 */
export function subscribeToCourseModelRequest(
  courseId: string,
  onData: (request: CourseModelRequest | null) => void,
  onError?: (message: string) => void,
): Unsubscribe {
  assertValidCourseId(courseId);

  return pollingSubscription<CourseModelRequest | null>({
    fetcher: () => fetchCourseModelRequest(courseId),
    onData,
    onError,
    shouldPoll: (request) => isActiveRequest(request),
  });
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
 * The guard moved into the database with the record. The backend inserts only
 * when no active request exists for the course, evaluated at write time, so a
 * professor double-clicking or two tabs racing still produce exactly one
 * request — the same guarantee the client-side transaction gave, enforced a layer
 * lower. A refused create comes back as 409.
 */
export async function createCourseModelRequest(
  courseId: string,
  approvedExampleCount: number,
): Promise<CourseModelRequest> {
  assertValidCourseId(courseId);

  try {
    const created = await dbApi.createModelRequest(
      courseId,
      Math.max(0, Math.trunc(approvedExampleCount)),
    );
    const parsed = parseCourseModelRequest(created);
    if (!parsed) {
      throw new Error('The backend returned an unreadable model request.');
    }
    return parsed;
  } catch (error) {
    if (error instanceof Error && 'status' in error && error.status === 409) {
      throw new DuplicateModelRequestError();
    }
    throw error;
  }
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
  await dbApi.updateModelRequest(courseId, patch);
}
