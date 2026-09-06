import { getJson, postJson, sendJson } from './httpClient';
import type { StaffRole } from './authApi';

/**
 * Administration: `/api/admin`.
 *
 * Accounts, roles, course memberships, privileged invitations and the audit
 * trail. Administrator-only on the backend; nothing here is reachable from a
 * professor session, and nothing here is rendered outside the admin area.
 */

export interface AdminUser {
  userId: string;
  email: string;
  displayName: string;
  role: StaffRole;
  disabled: boolean;
  createdAt?: string | null;
  lastLoginAt?: string | null;
  courseIds: string[];
}

export interface AdminUserList {
  count: number;
  users: AdminUser[];
}

export function listUsers(): Promise<AdminUserList> {
  return getJson<AdminUserList>('/admin/users', 'Could not load accounts.');
}

export interface UserPatch {
  disabled?: boolean;
  role?: StaffRole;
}

export function updateUser(userId: string, patch: UserPatch): Promise<AdminUser> {
  return sendJson<AdminUser>(
    'PATCH',
    `/admin/users/${encodeURIComponent(userId)}`,
    patch,
    'Could not update the account.',
  );
}

export interface MembershipResult {
  userId: string;
  courseId: string;
  member: boolean;
}

export function addMembership(userId: string, courseId: string): Promise<MembershipResult> {
  return sendJson<MembershipResult>(
    'PUT',
    `/admin/users/${encodeURIComponent(userId)}/courses/${encodeURIComponent(courseId)}`,
    undefined,
    'Could not add the course.',
  );
}

export function removeMembership(userId: string, courseId: string): Promise<MembershipResult> {
  return sendJson<MembershipResult>(
    'DELETE',
    `/admin/users/${encodeURIComponent(userId)}/courses/${encodeURIComponent(courseId)}`,
    undefined,
    'Could not remove the course.',
  );
}

export type InvitationKind = 'professor' | 'admin' | 'reset';
export type InvitationStatus = 'active' | 'revoked' | 'expired' | 'used';

export interface AdminInvitation {
  invitationId: string;
  kind: InvitationKind;
  status: InvitationStatus;
  createdAt: string;
  expiresAt?: string | null;
  label?: string | null;
  courseIds: string[];
  createdByName?: string | null;
  acceptedAt?: string | null;
  revokedAt?: string | null;
  targetUserId?: string | null;
}

/** The creating response, the only one that ever carries the token. */
export interface CreatedInvitation extends AdminInvitation {
  token: string;
  /** `/invite/<token>`; the page composes the full link from its own origin. */
  path: string;
}

export interface CreateInvitationBody {
  kind: 'professor' | 'admin';
  courseIds?: string[];
  expiresInHours?: number;
  label?: string;
}

export function createInvitation(body: CreateInvitationBody): Promise<CreatedInvitation> {
  return postJson<CreatedInvitation>('/admin/invitations', body, 'Could not create the invitation.');
}

export interface AdminInvitationList {
  count: number;
  invitations: AdminInvitation[];
}

export function listInvitations(status?: InvitationStatus): Promise<AdminInvitationList> {
  const query = status ? `?status=${encodeURIComponent(status)}` : '';
  return getJson<AdminInvitationList>(`/admin/invitations${query}`, 'Could not load invitations.');
}

export function revokeInvitation(invitationId: string): Promise<AdminInvitation> {
  return sendJson<AdminInvitation>(
    'POST',
    `/admin/invitations/${encodeURIComponent(invitationId)}/revoke`,
    undefined,
    'Could not revoke the invitation.',
  );
}

export function createResetInvite(userId: string): Promise<CreatedInvitation> {
  return sendJson<CreatedInvitation>(
    'POST',
    `/admin/users/${encodeURIComponent(userId)}/reset-invite`,
    undefined,
    'Could not create a reset link.',
  );
}

export interface AuditAction {
  actionId: number;
  action: string;
  createdAt: string;
  actorUserId?: string;
  actorRole?: string;
  actorName?: string;
  actorEmail?: string;
  targetKind?: string;
  targetId?: string;
  courseId?: string;
  detail?: Record<string, unknown>;
}

export interface AuditList {
  count: number;
  actions: AuditAction[];
}

export function listAudit(limit = 100, before?: number): Promise<AuditList> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (before !== undefined) {
    params.set('before', String(before));
  }
  return getJson<AuditList>(`/admin/audit?${params.toString()}`, 'Could not load the audit trail.');
}

/** The full invitation link for a token path, from the origin the admin is on. */
export function invitationLinkUrl(origin: string, path: string): string {
  return `${origin.replace(/\/$/, '')}${path}`;
}
