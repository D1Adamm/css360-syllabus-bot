import { Navigate, useLocation, useParams } from 'react-router-dom';
import { useSession } from '../context/SessionContext';
import { isAnonymous, roleForSession } from '../context/session';
import { DEFAULT_COURSE_ID, isValidCourseId } from '../lib/courseId';
import {
  adminCourseExamplesPath,
  adminCoursePath,
  professorCourseHomePath,
  professorCoursePath,
  joinPath,
  roleHomePath,
  studentCourseHomePath,
  studentCoursePath,
  type ProfessorCourseSegment,
  type StudentCourseSegment,
} from '../lib/roleRoutes';

/**
 * Redirects from the pre-role URL structure.
 *
 * Old links, bookmarks and anything already shared with a class must keep
 * working. Every redirect preserves the query string, and every one of them
 * resolves in a single hop rather than bouncing through an intermediate URL.
 */

/**
 * `/` — send whoever is signed in to their landing page.
 *
 * A browser with nothing remembered goes to the join page, which also links
 * staff to sign-in. A student goes to their course list, which opens their
 * course directly when there is only one.
 */
export function RoleLanding() {
  const { state, session } = useSession();
  if (state.status === 'loading') {
    return (
      <p className="ui-text-muted" role="status" aria-live="polite">
        Checking your session…
      </p>
    );
  }
  // A browser with no staff session and no joined course is a new student:
  // the class-code page is where they start. Staff sign in from there.
  if (isAnonymous(session)) {
    return <Navigate to={joinPath()} replace />;
  }
  // A student with one remembered course goes straight back into it; with
  // several, to the course list. The list itself never redirects, so "My
  // courses" always reaches "Join another course".
  if (!session.user && session.participants.length === 1) {
    return <Navigate to={studentCourseHomePath(session.participants[0].courseId)} replace />;
  }
  return <Navigate to={roleHomePath(roleForSession(session))} replace />;
}

interface CourseRedirectProps {
  /** Where this segment lives for each role. */
  student?: StudentCourseSegment | 'home';
  professor?: ProfessorCourseSegment | 'home';
  admin?: 'course' | 'examples';
}

/**
 * Redirect a `/course/:courseId/...` URL into the right role tree.
 *
 * Some old segments map to exactly one role — `compare` was only ever a student
 * page, `review` only a professor one — and those go there regardless of who
 * is signed in. `home` and `syllabus` existed for everyone, so they follow the
 * session's role.
 */
export function LegacyCourseRedirect({
  student,
  professor,
  admin,
}: CourseRedirectProps) {
  const { courseId } = useParams<{ courseId: string }>();
  const { search } = useLocation();
  const { session } = useSession();
  const role = roleForSession(session);

  // Preserve the original validation behaviour: an unusable id is not a
  // redirect target, it is an invalid course.
  if (!isValidCourseId(courseId)) {
    return <Navigate to="/not-found" replace />;
  }

  const targets: Record<string, string | null> = {
    student: student
      ? student === 'home'
        ? studentCourseHomePath(courseId)
        : studentCoursePath(courseId, student, search)
      : null,
    professor: professor
      ? professor === 'home'
        ? professorCourseHomePath(courseId)
        : professorCoursePath(courseId, professor, search)
      : null,
    admin: admin
      ? admin === 'examples'
        ? adminCourseExamplesPath(courseId)
        : adminCoursePath(courseId)
      : null,
  };

  // Prefer the current role's destination, then any role this segment supports.
  const destination =
    targets[role] ?? targets.student ?? targets.professor ?? targets.admin;

  return <Navigate to={destination ?? roleHomePath(role)} replace />;
}

/**
 * Redirect the oldest flat URLs (`/compare`, `/review`, …), which implicitly
 * referred to the default course.
 */
export function LegacyFlatRedirect(props: CourseRedirectProps) {
  const { search } = useLocation();
  const { session } = useSession();
  const role = roleForSession(session);

  const targets: Record<string, string | null> = {
    student: props.student
      ? props.student === 'home'
        ? studentCourseHomePath(DEFAULT_COURSE_ID)
        : studentCoursePath(DEFAULT_COURSE_ID, props.student, search)
      : null,
    professor: props.professor
      ? props.professor === 'home'
        ? professorCourseHomePath(DEFAULT_COURSE_ID)
        : professorCoursePath(DEFAULT_COURSE_ID, props.professor, search)
      : null,
    admin: props.admin
      ? props.admin === 'examples'
        ? adminCourseExamplesPath(DEFAULT_COURSE_ID)
        : adminCoursePath(DEFAULT_COURSE_ID)
      : null,
  };

  const destination =
    targets[role] ?? targets.student ?? targets.professor ?? targets.admin;

  return <Navigate to={destination ?? roleHomePath(role)} replace />;
}
