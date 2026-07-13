/**
 * Temporary default course id for future multi-course migration.
 * Existing CSS 360 data remains at global `seedExamples` / `evaluations`
 * until an explicit migration step moves it under courses/{DEFAULT_COURSE_ID}.
 */
export const DEFAULT_COURSE_ID = 'css360-default';

const COURSE_ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

/**
 * A valid courseId is non-empty, lowercase letters/numbers/hyphens only,
 * does not begin or end with a hyphen, and must not contain path-unsafe
 * characters (slashes, dots, brackets, dollar signs, traversal patterns).
 */
export function isValidCourseId(courseId: unknown): courseId is string {
  if (typeof courseId !== 'string' || courseId.length === 0) {
    return false;
  }

  if (/[./\\[\]$]/.test(courseId) || courseId.includes('..')) {
    return false;
  }

  if (courseId.startsWith('-') || courseId.endsWith('-')) {
    return false;
  }

  return COURSE_ID_PATTERN.test(courseId);
}

export function assertValidCourseId(courseId: unknown): asserts courseId is string {
  if (!isValidCourseId(courseId)) {
    throw new Error(
      `Invalid courseId "${String(courseId)}": must be non-empty, use lowercase letters, numbers, and hyphens only, and must not begin/end with a hyphen or contain path-unsafe characters.`,
    );
  }
}
