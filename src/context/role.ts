export const ROLES = ['student', 'professor', 'admin'] as const;

export type Role = (typeof ROLES)[number];

export const DEFAULT_ROLE: Role = 'student';

const STORAGE_KEY = 'sml.devRole';

export function isRole(value: unknown): value is Role {
  return typeof value === 'string' && (ROLES as readonly string[]).includes(value);
}

/**
 * Reads the remembered development role.
 *
 * This is a convenience for whoever is working on the prototype — it is NOT
 * authentication and NOT authorization. Nothing in the application trusts it
 * for access control, and every route remains reachable by URL whatever it
 * returns. Real users will never choose their own role.
 */
export function readStoredRole(): Role {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return isRole(stored) ? stored : DEFAULT_ROLE;
  } catch {
    // Private browsing or a disabled storage API is not an error here.
    return DEFAULT_ROLE;
  }
}

export function writeStoredRole(role: Role): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, role);
  } catch {
    // Remembering the choice is best-effort.
  }
}
