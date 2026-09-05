export const ROLES = ['student', 'professor', 'admin'] as const;

/**
 * The three audiences the interface is written for.
 *
 * A vocabulary for chrome and navigation, not an identity: which role a
 * browser is shown comes from the URL and from the session the backend
 * reported (`context/session.ts`), never from anything stored locally.
 */
export type Role = (typeof ROLES)[number];

export const DEFAULT_ROLE: Role = 'student';

export function isRole(value: unknown): value is Role {
  return typeof value === 'string' && (ROLES as readonly string[]).includes(value);
}
