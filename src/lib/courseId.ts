/**
 * Reserved course id used only by legacy flat-route redirects
 * (`/home`, `/compare`, … → `/course/css360-default/...`).
 * Live pages always take courseId from `/course/:courseId/...`.
 */
export const DEFAULT_COURSE_ID = 'css360-default';

const COURSE_ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const COURSE_ID_SUFFIX_ALPHABET = 'abcdefghijklmnopqrstuvwxyz0123456789';

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

/** Slugify a display label into a courseId-safe segment. */
export function slugifyCourseIdPart(value: string): string {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-+|-+$/g, '');
}

export function generateCourseIdSuffix(length = 4): string {
  const bytes = new Uint8Array(length);
  crypto.getRandomValues(bytes);

  let suffix = '';
  for (let index = 0; index < length; index += 1) {
    suffix += COURSE_ID_SUFFIX_ALPHABET[bytes[index]! % COURSE_ID_SUFFIX_ALPHABET.length];
  }
  return suffix;
}

/**
 * Build a safe unique-looking course id from course name/code and term.
 * Example: generateCourseId('CSS 430', 'Summer 2026') → css430-summer-2026-a82f
 */
export function generateCourseId(courseName: string, term: string): string {
  const namePart = slugifyCourseIdPart(courseName);
  const termPart = slugifyCourseIdPart(term);
  const base = [namePart, termPart].filter((part) => part.length > 0).join('-') || 'course';
  const courseId = `${base}-${generateCourseIdSuffix(4)}`;
  assertValidCourseId(courseId);
  return courseId;
}
