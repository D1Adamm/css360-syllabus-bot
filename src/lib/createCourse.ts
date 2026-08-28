import type { CourseMetadata } from '../types';
import { generateCourseId } from './courseId';
import { courseExists, createCourseMetadata } from './coursesDb';

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

/**
 * Generate a unique courseId, then save CourseMetadata as its `courses` row.
 * Regenerates the random suffix if courseExists reports a collision.
 */
export async function createCourse(input: CreateCourseInput): Promise<CreateCourseResult> {
  const metadata = buildCourseMetadata(input);

  for (let attempt = 0; attempt < MAX_COURSE_ID_ATTEMPTS; attempt += 1) {
    const courseId = generateCourseId(metadata.name, metadata.term);
    const exists = await courseExists(courseId);

    if (!exists) {
      await createCourseMetadata(courseId, metadata);
      return { courseId, metadata };
    }
  }

  throw new Error(
    'Could not allocate a unique course id after several attempts. Please try again.',
  );
}
