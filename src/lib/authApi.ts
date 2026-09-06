import { getJson, postJson, sendJson } from './httpClient';

/**
 * Sessions: `/api/auth`.
 *
 * The only client that creates a session. `fetchSession` is what the session
 * provider asks on load; `login` and `joinCourse` set cookies the browser
 * keeps; `logout` ends everything this browser holds. Nothing here reads or
 * stores a token — the cookies are HttpOnly and this code never sees them.
 */

export type StaffRole = 'admin' | 'professor';

export interface UserSession {
  userId: string;
  email: string;
  displayName: string;
  role: StaffRole;
  /** Courses this professor holds an instructor membership in. Empty for admins. */
  courseIds: string[];
}

export interface ParticipantSession {
  /** The one course this anonymous student joined. Nothing else is exposed. */
  courseId: string;
}

export interface Session {
  user: UserSession | null;
  participant: ParticipantSession | null;
}

export const ANONYMOUS_SESSION: Session = { user: null, participant: null };

export function fetchSession(): Promise<Session> {
  return getJson<Session>('/auth/session', 'Could not check who is signed in.');
}

export function login(email: string, password: string): Promise<void> {
  return sendJson<void>(
    'POST',
    '/auth/login',
    { email, password },
    'The email address or password is not correct.',
  );
}

export function logout(): Promise<void> {
  return sendJson<void>('POST', '/auth/logout', undefined, 'Could not sign out.');
}

export interface JoinResult {
  courseId: string;
  courseName?: string | null;
  /** True when this browser had already joined the course with this code. */
  alreadyJoined: boolean;
}

export function joinCourse(code: string): Promise<JoinResult> {
  return postJson<JoinResult>(
    '/auth/join',
    { code },
    "That code isn't valid right now. Check it with your instructor and try again.",
  );
}

export interface InvitationCourse {
  courseId: string;
  name?: string | null;
}

export interface InvitationPreview {
  kind: 'professor' | 'admin' | 'reset';
  label?: string | null;
  expiresAt?: string | null;
  courses: InvitationCourse[];
  targetEmail?: string | null;
}

export function previewInvitation(token: string): Promise<InvitationPreview> {
  return getJson<InvitationPreview>(
    `/auth/invitations/${encodeURIComponent(token)}`,
    'This invitation link is not valid.',
  );
}

export interface AcceptInvitationBody {
  email?: string;
  displayName?: string;
  password: string;
}

export function acceptInvitation(
  token: string,
  body: AcceptInvitationBody,
): Promise<Session> {
  return postJson<Session>(
    `/auth/invitations/${encodeURIComponent(token)}/accept`,
    body,
    'The invitation could not be accepted.',
  );
}

export function changePassword(currentPassword: string, newPassword: string): Promise<void> {
  return sendJson<void>(
    'POST',
    '/auth/password',
    { currentPassword, newPassword },
    'The password could not be changed.',
  );
}
