import { getJson, postJson, sendJson } from './httpClient';

/**
 * Classroom codes for one course: `/api/courses/{courseId}/student-invites`.
 *
 * Course-staff only on the backend; a professor reaches these for the courses
 * they hold a membership in and an administrator for any course. The code is
 * returned on every read because it has to be shown on the page again — it is
 * the one credential in the system that is not a secret from its owner.
 */

export type StudentInviteStatus = 'active' | 'revoked' | 'expired' | 'used';

export interface StudentInvite {
  invitationId: string;
  courseId: string;
  code: string;
  status: StudentInviteStatus;
  createdAt: string;
  useCount: number;
  participantCount: number;
  label?: string | null;
  expiresAt?: string | null;
  revokedAt?: string | null;
  createdByName?: string | null;
}

export interface StudentInviteList {
  courseId: string;
  count: number;
  invites: StudentInvite[];
}

function base(courseId: string): string {
  return `/courses/${encodeURIComponent(courseId)}/student-invites`;
}

export function listStudentInvites(courseId: string): Promise<StudentInviteList> {
  return getJson<StudentInviteList>(base(courseId), 'Could not load the class codes.');
}

export interface CreateStudentInviteBody {
  label?: string;
  /** ISO 8601. Omit for a code that lasts until it is revoked. */
  expiresAt?: string;
  /** Retire every other live code for the course in the same step. */
  replaceExisting?: boolean;
}

export function createStudentInvite(
  courseId: string,
  body: CreateStudentInviteBody = {},
): Promise<StudentInvite> {
  return postJson<StudentInvite>(base(courseId), body, 'Could not create a class code.');
}

export function revokeStudentInvite(
  courseId: string,
  invitationId: string,
): Promise<StudentInvite> {
  return sendJson<StudentInvite>(
    'POST',
    `${base(courseId)}/${encodeURIComponent(invitationId)}/revoke`,
    undefined,
    'Could not revoke the class code.',
  );
}

/** The page students type the code into, and the direct link for a code. */
export function joinPageUrl(origin: string): string {
  return `${origin.replace(/\/$/, '')}/join`;
}

export function joinLinkUrl(origin: string, code: string): string {
  return `${joinPageUrl(origin)}/${encodeURIComponent(code)}`;
}
