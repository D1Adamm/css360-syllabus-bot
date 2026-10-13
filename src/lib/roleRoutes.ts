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

/** Sign-in for professors and administrators. */
export function loginPath(): string {
  return '/login';
}

/** The student join page, optionally with the class code filled in. */
export function joinPath(code?: string): string {
  return code ? `/join/${encodeURIComponent(code)}` : '/join';
}

/** Accepting a professor, administrator or reset invitation. */
export function invitePath(token: string): string {
  return `/invite/${encodeURIComponent(token)}`;
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

/** The professor review workflow, mounted in the admin tree for any course. */
export function adminCourseReviewPath(courseId: string): string {
  return `/admin/courses/${courseId}/review`;
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

/** The two staff interfaces an administrator can switch between. */
export type StaffView = 'admin' | 'professor';

/**
 * Where switching to `view` from `pathname` lands, keeping the course in view.
 *
 * Course pages map to the same course's page in the other interface (the
 * review workflow to its counterpart, anything else to the course's home), the
 * two course lists map to each other, and everything else goes to the view's
 * landing page. Only the URL changes: what the account may do is the session's
 * role, which the route guards and the backend check as before.
 */
export function staffViewPath(view: StaffView, pathname: string): string {
  const courseId = getCourseIdFromRolePathname(pathname);
  const fromReview =
    /^\/professor\/course\/[^/]+\/examples(\/|$)/.test(pathname) ||
    /^\/admin\/courses\/[^/]+\/review(\/|$)/.test(pathname);

  if (view === 'professor') {
    if (courseId) {
      return fromReview
        ? professorCoursePath(courseId, 'examples')
        : professorCourseHomePath(courseId);
    }
    return '/professor/courses';
  }

  if (courseId) {
    return fromReview ? adminCourseReviewPath(courseId) : adminCoursePath(courseId);
  }
  if (pathname === '/professor/courses' || pathname.startsWith('/professor/courses/')) {
    return '/admin/courses';
  }
  return roleHomePath('admin');
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
