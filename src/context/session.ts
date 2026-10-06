import type { Role } from './role';
import type { Session } from '../lib/authApi';

export type { Session } from '../lib/authApi';

/**
 * Reading a session.
 *
 * Pure functions over what `/api/auth/session` returned. They decide which
 * chrome to draw and which route guard lets a page render — presentation and
 * navigation only. Every one of these questions is asked again, and answered
 * authoritatively, by the backend on every request; nothing here grants
 * anything.
 */

export type SessionState =
  | { status: 'loading' }
  | { status: 'ready'; session: Session }
  /** The session could not be checked. Says nothing about who is signed in. */
  | { status: 'unavailable'; message: string };

/**
 * How long one session check may go unanswered, and how often to try.
 *
 * Without a limit, a check that never settles kept a new tab on "Checking your
 * session…" for good. That happens when the browser cannot send it: over
 * HTTP/1.1 a browser opens at most six connections per host, shared by every
 * tab, and one Compare run in another tab can hold them all for as long as its
 * generations take. Stopping the request takes it out of the browser's queue;
 * trying again later gets it a connection once one is free. Only a failure to
 * get any answer is retried: an answer from the backend, even an error, is
 * final.
 */
export const SESSION_CHECK_TIMEOUT_MS = 10_000;
export const SESSION_CHECK_RETRY_DELAYS_MS = [1_000, 3_000] as const;

export function isAnonymous(session: Session): boolean {
  return session.user === null && session.participants.length === 0;
}

export function isStaff(session: Session): boolean {
  return session.user !== null;
}

export function isAdmin(session: Session): boolean {
  return session.user?.role === 'admin';
}

/** The chrome to draw when the URL does not say: admin, professor, or student. */
export function roleForSession(session: Session): Role {
  if (session.user?.role === 'admin') {
    return 'admin';
  }
  if (session.user?.role === 'professor') {
    return 'professor';
  }
  return 'student';
}

/** Act on a course as its staff: administrators always, professors by membership. */
export function canStaffCourse(session: Session, courseId: string): boolean {
  if (!session.user) {
    return false;
  }
  return session.user.role === 'admin' || session.user.courseIds.includes(courseId);
}

/** The courses this browser joined with a class code. */
export function joinedCourseIds(session: Session): string[] {
  return session.participants.map((participant) => participant.courseId);
}

export function isParticipantOf(session: Session, courseId: string): boolean {
  return session.participants.some((participant) => participant.courseId === courseId);
}

/** Open a course's student pages: its staff, or the participant who joined it. */
export function canAccessCourse(session: Session, courseId: string): boolean {
  return canStaffCourse(session, courseId) || isParticipantOf(session, courseId);
}

/**
 * An administrator previewing a course's student pages.
 *
 * Administrators never redeem a classroom code, so on a course's student pages
 * there is no participant to attribute a rating to. The pages run for real
 * against that course — the same four generation requests a student makes —
 * and the last step changes: nothing submitted is saved, and the backend's
 * preview route is what says so. An administrator who did join the course
 * holds a participant there and is not previewing; neither is a professor,
 * whose way of trying the flow is their own classroom code, unchanged.
 */
export function isAdminPreview(session: Session, courseId: string): boolean {
  return isAdmin(session) && !isParticipantOf(session, courseId);
}
