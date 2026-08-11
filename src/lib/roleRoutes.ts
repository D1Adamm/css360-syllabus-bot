import type { Role } from '../context/role';
import { isValidCourseId } from './courseId';

/**
 * Path builders for the role-scoped route trees.
 *
 * The legacy `coursePagePath` in `courseRoutes.ts` still builds `/course/:id/*`
 * URLs and is kept so old links and its tests keep working; those URLs now
 * redirect here.
 */

export const STUDENT_COURSE_SEGMENTS = [
  'syllabus',
  'contribute',
  'compare',
  'evaluate',
] as const;

export const PROFESSOR_COURSE_SEGMENTS = [
  'syllabus',
  'examples',
  'model',
  'results',
  'invite',
] as const;

export type StudentCourseSegment = (typeof STUDENT_COURSE_SEGMENTS)[number];
export type ProfessorCourseSegment = (typeof PROFESSOR_COURSE_SEGMENTS)[number];

function withSearch(path: string, search: string): string {
  if (!search) {
    return path;
  }
  return `${path}${search.startsWith('?') ? search : `?${search}`}`;
}

/** Where each role lands when it has no more specific destination. */
export function roleHomePath(role: Role): string {
  switch (role) {
    case 'professor':
      return '/professor/courses';
    case 'admin':
      return '/admin';
    default:
      return '/student';
  }
}

export function studentCourseHomePath(courseId: string): string {
  return `/student/course/${courseId}`;
}

export function studentCoursePath(
  courseId: string,
  segment: StudentCourseSegment,
  search = '',
): string {
  return withSearch(`/student/course/${courseId}/${segment}`, search);
}

export function professorCourseHomePath(courseId: string): string {
  return `/professor/course/${courseId}`;
}

export function professorCoursePath(
  courseId: string,
  segment: ProfessorCourseSegment,
  search = '',
): string {
  return withSearch(`/professor/course/${courseId}/${segment}`, search);
}

export function adminCoursePath(courseId: string): string {
  return `/admin/courses/${courseId}`;
}

export function adminCourseExamplesPath(courseId: string): string {
  return `/admin/courses/${courseId}/examples`;
}

/**
 * Which role area a pathname belongs to, or `null` for role-neutral routes.
 *
 * The URL — not the remembered development role — decides which shell and
 * navigation render. That keeps a deep link into, say, a professor page from
 * showing student navigation just because of a leftover localStorage value.
 */
export function getRoleAreaFromPathname(pathname: string): Role | null {
  if (pathname === '/student' || pathname.startsWith('/student/')) {
    return 'student';
  }
  if (pathname === '/professor' || pathname.startsWith('/professor/')) {
    return 'professor';
  }
  if (pathname === '/admin' || pathname.startsWith('/admin/')) {
    return 'admin';
  }
  return null;
}

const COURSE_PATH_PATTERNS = [
  /^\/student\/course\/([^/]+)/,
  /^\/professor\/course\/([^/]+)/,
  /^\/admin\/courses\/([^/]+)/,
];

/** Resolve a courseId from any role-scoped course URL. */
export function getCourseIdFromRolePathname(pathname: string): string | null {
  for (const pattern of COURSE_PATH_PATTERNS) {
    const match = pathname.match(pattern);
    if (match) {
      const courseId = match[1];
      return isValidCourseId(courseId) ? courseId : null;
    }
  }
  return null;
}
