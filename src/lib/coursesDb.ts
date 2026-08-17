import type { CourseMetadata, SyllabusStatus } from '../types';
import { assertValidCourseId } from './courseId';
import * as dbApi from './dbApi';
import { pollingSubscription, type Unsubscribe } from './pollingSubscription';

/**
 * Course metadata, now read from PostgreSQL through FastAPI.
 *
 * The module keeps the shape it had when it spoke to Firebase —
 * `subscribeToCourses(onData, onError)` returning an unsubscribe — so the hooks
 * and pages above it did not have to change. What changed is underneath: a
 * realtime listener became a fetch, because a course list only changes when
 * someone on this system creates or edits a course.
 *
 * `isCourseMetadata` still runs over what the API returns. The backend is ours
 * and its response model is typed, but this is the guard that decides whether a
 * professor sees their course or a blank picker, and it costs nothing to keep
 * validating at the boundary.
 */

/** Course picker list item: metadata plus the course id. */
export interface CourseListItem {
  courseId: string;
  metadata: CourseMetadata;
}

const SYLLABUS_STATUSES: readonly SyllabusStatus[] = [
  'none',
  'not_uploaded',
  'uploaded',
  'extracted',
  'indexed',
  'upload_failed',
  'index_failed',
  'processing',
  'ready',
  'error',
];

function isSyllabusStatus(value: unknown): value is SyllabusStatus {
  return typeof value === 'string' && SYLLABUS_STATUSES.includes(value as SyllabusStatus);
}

export function isCourseMetadata(value: unknown): value is CourseMetadata {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const record = value as Record<string, unknown>;

  return (
    typeof record.name === 'string' &&
    typeof record.title === 'string' &&
    typeof record.term === 'string' &&
    typeof record.instructorName === 'string' &&
    typeof record.createdAt === 'string' &&
    isSyllabusStatus(record.syllabusStatus) &&
    (record.syllabusFileName === null || typeof record.syllabusFileName === 'string') &&
    (record.syllabusType === null || typeof record.syllabusType === 'string') &&
    typeof record.chunkCount === 'number' &&
    Number.isFinite(record.chunkCount)
  );
}

export function sortCoursesNewestFirst(courses: CourseListItem[]): CourseListItem[] {
  return [...courses].sort((left, right) => {
    const leftTime = Date.parse(left.metadata.createdAt);
    const rightTime = Date.parse(right.metadata.createdAt);

    if (Number.isNaN(leftTime) && Number.isNaN(rightTime)) {
      return left.courseId.localeCompare(right.courseId);
    }
    if (Number.isNaN(leftTime)) {
      return 1;
    }
    if (Number.isNaN(rightTime)) {
      return -1;
    }
    if (rightTime !== leftTime) {
      return rightTime - leftTime;
    }

    return left.courseId.localeCompare(right.courseId);
  });
}

/** Keep only entries the UI can actually render, newest first. */
export function parseCourseList(courses: dbApi.DbCourseRecord[]): CourseListItem[] {
  const items: CourseListItem[] = [];

  for (const entry of courses) {
    if (typeof entry?.courseId !== 'string' || !isCourseMetadata(entry.metadata)) {
      continue;
    }
    items.push({ courseId: entry.courseId, metadata: entry.metadata });
  }

  return sortCoursesNewestFirst(items);
}

export async function createCourseMetadata(
  courseId: string,
  metadata: CourseMetadata,
): Promise<void> {
  assertValidCourseId(courseId);

  await dbApi.createCourse({
    courseId,
    ...metadata,
    createdAt: metadata.createdAt || new Date().toISOString(),
    syllabusFileName: metadata.syllabusFileName ?? null,
    syllabusType: metadata.syllabusType ?? null,
    chunkCount: metadata.chunkCount ?? 0,
  });
}

export async function getCourseMetadata(courseId: string): Promise<CourseMetadata | null> {
  assertValidCourseId(courseId);

  try {
    const course = await dbApi.getCourse(courseId);
    return isCourseMetadata(course.metadata) ? course.metadata : null;
  } catch (error) {
    // A missing course is an answer, not a failure. Anything else is a real
    // read error and must not be reported as "no such course".
    if (error instanceof Error && 'status' in error && error.status === 404) {
      return null;
    }
    throw error;
  }
}

/**
 * Statuses that mean the starter-generation job is still working.
 *
 * The job runs in the backend and writes its progress where this metadata comes
 * from, so a professor watching the page has no other way to learn it finished.
 * These two are the only reason this subscription ever polls.
 */
const ACTIVE_STARTER_STATUSES = new Set(['queued', 'generating']);

function starterGenerationIsRunning(metadata: CourseMetadata | null): boolean {
  const status = metadata?.starterSeedGeneration?.status;
  return typeof status === 'string' && ACTIVE_STARTER_STATUSES.has(status);
}

export function subscribeToCourseMetadata(
  courseId: string,
  onData: (metadata: CourseMetadata | null) => void,
  onError?: (message: string) => void,
): Unsubscribe {
  assertValidCourseId(courseId);

  return pollingSubscription<CourseMetadata | null>({
    fetcher: () => getCourseMetadata(courseId),
    onData,
    onError,
    // Poll only while examples are actually being generated. `ready`,
    // `partial`, and `failed` are all terminal, and a course sitting in one of
    // them is static data that would otherwise be re-fetched forever.
    shouldPoll: starterGenerationIsRunning,
  });
}

export async function courseExists(courseId: string): Promise<boolean> {
  assertValidCourseId(courseId);
  return (await getCourseMetadata(courseId)) !== null;
}

export async function updateCourseMetadata(
  courseId: string,
  updates: Partial<CourseMetadata>,
): Promise<void> {
  assertValidCourseId(courseId);
  await dbApi.updateCourse(courseId, updates);
}

/**
 * All courses for the picker.
 *
 * Fetched once per mount rather than watched. A course appearing while a
 * professor stares at the list is not a scenario worth a polling timer; the
 * pages that create one navigate straight to it.
 */
export function subscribeToCourses(
  onData: (courses: CourseListItem[]) => void,
  onError?: (message: string) => void,
): Unsubscribe {
  return pollingSubscription<CourseListItem[]>({
    fetcher: async () => parseCourseList((await dbApi.listCourses()).courses),
    onData,
    onError,
  });
}
