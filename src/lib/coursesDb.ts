import {
  get,
  onValue,
  ref,
  set,
  update,
  type Unsubscribe,
} from 'firebase/database';
import type { CourseMetadata, SyllabusStatus } from '../types';
import { assertValidCourseId, isValidCourseId } from './courseId';
import { COURSES_ROOT_PATH, getCourseMetadataPath } from './coursePaths';
import { database } from './firebase';

/** Course picker list item: metadata plus the Firebase child key as courseId. */
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

export function getCourseMetadataRef(courseId: string) {
  return ref(database, getCourseMetadataPath(courseId));
}

export async function createCourseMetadata(
  courseId: string,
  metadata: CourseMetadata,
): Promise<void> {
  assertValidCourseId(courseId);

  const stored: CourseMetadata = {
    ...metadata,
    createdAt: metadata.createdAt || new Date().toISOString(),
    syllabusFileName: metadata.syllabusFileName ?? null,
    syllabusType: metadata.syllabusType ?? null,
    chunkCount: metadata.chunkCount ?? 0,
  };

  await set(getCourseMetadataRef(courseId), stored);
}

export async function getCourseMetadata(courseId: string): Promise<CourseMetadata | null> {
  assertValidCourseId(courseId);

  const snapshot = await get(getCourseMetadataRef(courseId));
  if (!snapshot.exists()) {
    return null;
  }

  const value = snapshot.val();
  return isCourseMetadata(value) ? value : null;
}

export function subscribeToCourseMetadata(
  courseId: string,
  onData: (metadata: CourseMetadata | null) => void,
  onError?: (message: string) => void,
): Unsubscribe {
  assertValidCourseId(courseId);

  return onValue(
    getCourseMetadataRef(courseId),
    (snapshot) => {
      if (!snapshot.exists()) {
        onData(null);
        return;
      }

      const value = snapshot.val();
      onData(isCourseMetadata(value) ? value : null);
    },
    (error) => {
      onError?.(error.message);
    },
  );
}

export async function updateCourseMetadata(
  courseId: string,
  updates: Partial<CourseMetadata>,
): Promise<void> {
  assertValidCourseId(courseId);
  await update(getCourseMetadataRef(courseId), updates);
}

export async function courseExists(courseId: string): Promise<boolean> {
  assertValidCourseId(courseId);
  const snapshot = await get(getCourseMetadataRef(courseId));
  return snapshot.exists();
}

export function getCoursesRef() {
  return ref(database, COURSES_ROOT_PATH);
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

/**
 * Transform a `courses` RTDB snapshot value into validated CourseListItem entries.
 * Only children with a valid courseId and metadata object are included.
 */
export function parseCoursesSnapshot(value: unknown): CourseListItem[] {
  if (!value || typeof value !== 'object') {
    return [];
  }

  const items: CourseListItem[] = [];

  for (const [courseId, courseNode] of Object.entries(value as Record<string, unknown>)) {
    if (!isValidCourseId(courseId) || !courseNode || typeof courseNode !== 'object') {
      continue;
    }

    const metadata = (courseNode as Record<string, unknown>).metadata;
    if (!isCourseMetadata(metadata)) {
      continue;
    }

    items.push({ courseId, metadata });
  }

  return sortCoursesNewestFirst(items);
}

/**
 * Subscribe to all courses under `courses` for the course picker.
 * Does not load seed examples or evaluations—only child metadata.
 */
export function subscribeToCourses(
  onData: (courses: CourseListItem[]) => void,
  onError?: (message: string) => void,
): Unsubscribe {
  return onValue(
    getCoursesRef(),
    (snapshot) => {
      if (!snapshot.exists()) {
        onData([]);
        return;
      }

      onData(parseCoursesSnapshot(snapshot.val()));
    },
    (error) => {
      onError?.(error.message);
    },
  );
}
