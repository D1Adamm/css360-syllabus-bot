import { Navigate, useLocation, useParams } from 'react-router-dom';
import { useSession } from '../../context/SessionContext';
import {
  canAccessCourse,
  canStaffCourse,
  isAdmin,
  isAnonymous,
  isStaff,
  type Session,
} from '../../context/session';
import { isValidCourseId } from '../../lib/courseId';
import { joinPath, loginPath } from '../../lib/roleRoutes';

/**
 * Route guards.
 *
 * Presentation only. Each one decides whether to render a page or send the
 * browser somewhere more useful — sign-in, the join page, or an explanation —
 * based on the cached session. The backend makes the same decision again for
 * every request the page issues, and the backend's answer is the one that
 * counts: a guard that let the wrong page render would show empty panels and
 * 403 banners, not data.
 */

type Verdict = 'allow' | 'sign-in' | 'join' | 'forbidden';

function Gate({
  decide,
  children,
}: {
  decide: (session: Session, courseId: string | undefined) => Verdict;
  children: React.ReactNode;
}) {
  const { state, session } = useSession();
  const location = useLocation();
  const { courseId } = useParams<{ courseId?: string }>();

  if (state.status === 'loading') {
    return (
      <p className="ui-text-muted" role="status" aria-live="polite">
        Checking your session…
      </p>
    );
  }

  // A course id that fails validation is the course route's problem to
  // explain; the guard must not redirect a typo to the sign-in page.
  const scopedCourse = courseId && isValidCourseId(courseId) ? courseId : undefined;
  const verdict = decide(session, scopedCourse);

  switch (verdict) {
    case 'allow':
      return <>{children}</>;
    case 'sign-in':
      return <Navigate to={loginPath()} replace state={{ from: location.pathname }} />;
    case 'join':
      return <Navigate to={joinPath()} replace state={{ from: location.pathname }} />;
    default:
      return <Navigate to="/forbidden" replace state={{ from: location.pathname }} />;
  }
}

/** Any professor or administrator. */
export function RequireStaff({ children }: { children: React.ReactNode }) {
  return <Gate decide={(session) => (isStaff(session) ? 'allow' : 'sign-in')}>{children}</Gate>;
}

/** Administrators only. */
export function RequireAdmin({ children }: { children: React.ReactNode }) {
  return (
    <Gate
      decide={(session) =>
        isAdmin(session) ? 'allow' : isStaff(session) ? 'forbidden' : 'sign-in'
      }
    >
      {children}
    </Gate>
  );
}

/** Course staff for the `:courseId` in the URL: admins, or professors assigned to it. */
export function RequireCourseStaff({ children }: { children: React.ReactNode }) {
  return (
    <Gate
      decide={(session, courseId) => {
        if (!isStaff(session)) {
          return 'sign-in';
        }
        if (!courseId) {
          return 'allow'; // an invalid id renders the invalid-course page below
        }
        return canStaffCourse(session, courseId) ? 'allow' : 'forbidden';
      }}
    >
      {children}
    </Gate>
  );
}

/**
 * The student pages of the `:courseId` in the URL: the participant who joined
 * it, or its staff walking through the student flow.
 */
export function RequireCourseParticipant({ children }: { children: React.ReactNode }) {
  return (
    <Gate
      decide={(session, courseId) => {
        if (!courseId) {
          return 'allow';
        }
        if (canAccessCourse(session, courseId)) {
          return 'allow';
        }
        return isAnonymous(session) ? 'join' : 'forbidden';
      }}
    >
      {children}
    </Gate>
  );
}

/** Anyone with a session at all — a participant or staff. */
export function RequireAnyPrincipal({ children }: { children: React.ReactNode }) {
  return (
    <Gate decide={(session) => (isAnonymous(session) ? 'join' : 'allow')}>{children}</Gate>
  );
}
