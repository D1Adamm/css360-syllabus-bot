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

export function isAnonymous(session: Session): boolean {
  return session.user === null && session.participant === null;
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

export function isParticipantOf(session: Session, courseId: string): boolean {
  return session.participant?.courseId === courseId;
}

/** Open a course's student pages: its staff, or the participant who joined it. */
export function canAccessCourse(session: Session, courseId: string): boolean {
  return canStaffCourse(session, courseId) || isParticipantOf(session, courseId);
}
