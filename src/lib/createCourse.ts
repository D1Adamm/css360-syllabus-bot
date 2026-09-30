import type { CourseMetadata } from '../types';
import { generateCourseId } from './courseId';
import { createCourseMetadata } from './coursesDb';
import { ApiError } from './httpClient';

const MAX_COURSE_ID_ATTEMPTS = 8;

export interface CreateCourseInput {
  name: string;
  title: string;
  term: string;
  instructorName?: string;
}

export interface CreateCourseResult {
  courseId: string;
  metadata: CourseMetadata;
}

function buildCourseMetadata(input: CreateCourseInput): CourseMetadata {
  return {
    name: input.name.trim(),
    title: input.title.trim(),
    term: input.term.trim(),
    instructorName: input.instructorName?.trim() ?? '',
    createdAt: new Date().toISOString(),
    syllabusStatus: 'not_uploaded',
    syllabusFileName: '',
    syllabusType: '',
    chunkCount: 0,
  };
}

function isIdCollision(error: unknown): boolean {
  return error instanceof ApiError && error.status === 409;
}

/**
 * Generate a courseId and save CourseMetadata as its `courses` row,
 * regenerating the random suffix when the backend answers 409 for a taken id.
 *
 * The database decides uniqueness. There is deliberately no read-before-create:
 * reading a course the caller does not staff yet is refused (403) for every
 * professor — it is how course creation broke for all non-admin staff — and
 * a check-then-insert would race another create anyway.
 */
export async function createCourse(input: CreateCourseInput): Promise<CreateCourseResult> {
  const metadata = buildCourseMetadata(input);

  for (let attempt = 0; attempt < MAX_COURSE_ID_ATTEMPTS; attempt += 1) {
    const courseId = generateCourseId(metadata.name, metadata.term);
    try {
      await createCourseMetadata(courseId, metadata);
      return { courseId, metadata };
    } catch (error) {
      if (!isIdCollision(error)) {
        throw error;
      }
    }
  }

  throw new Error(
    'Could not allocate a unique course id after several attempts. Please try again.',
  );
}
