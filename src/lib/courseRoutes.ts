import { DEFAULT_COURSE_ID, isValidCourseId } from './courseId';

export const COURSE_PAGE_SEGMENTS = [
  'home',
  'syllabus',
  'seeds',
  'review',
  'dataset',
  'compare',
  'evaluate',
  'results',
] as const;

export type CoursePageSegment = (typeof COURSE_PAGE_SEGMENTS)[number];

export function coursePagePath(
  courseId: string,
  segment: CoursePageSegment,
  search = '',
): string {
  const normalizedSearch = search && !search.startsWith('?') ? `?${search}` : search;
  return `/course/${courseId}/${segment}${normalizedSearch}`;
}

export function defaultCoursePagePath(segment: CoursePageSegment, search = ''): string {
  return coursePagePath(DEFAULT_COURSE_ID, segment, search);
}

/**
 * Resolve courseId from any course-scoped pathname.
 *
 * Matches the legacy `/course/{courseId}/...` shape as well as the role-scoped
 * `/student/course/...`, `/professor/course/...` and `/admin/courses/...`
 * trees, so shared chrome can find the active course during the migration.
 */
export function getCourseIdFromPathname(pathname: string): string | null {
  const match = pathname.match(
    /^(?:\/(?:student|professor)\/course|\/admin\/courses|\/course)\/([^/]+)/,
  );
  if (!match) {
    return null;
  }

  const courseId = match[1];
  return isValidCourseId(courseId) ? courseId : null;
}
