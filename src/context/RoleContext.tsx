import { createContext, useCallback, useContext, useMemo, useState } from 'react';
import { readStoredRole, writeStoredRole, type Role } from './role';

export type { Role } from './role';

interface RoleContextValue {
  /** The remembered development role. Drives `/` and the role switcher only. */
  role: Role;
  setRole: (role: Role) => void;
}

const RoleContext = createContext<RoleContextValue | null>(null);

/**
 * Holds the development role selection.
 *
 * Not an auth boundary: the shell reads the role area from the URL, and no
 * route is gated on this value. It exists so the prototype can be walked
 * through as each audience before sign-in exists.
 */
export function RoleProvider({
  children,
  initialRole,
}: {
  children: React.ReactNode;
  /** Test seam. Omit in the application. */
  initialRole?: Role;
}) {
  const [role, setRoleState] = useState<Role>(() => initialRole ?? readStoredRole());

  const setRole = useCallback((next: Role) => {
    setRoleState(next);
    writeStoredRole(next);
  }, []);

  const value = useMemo(() => ({ role, setRole }), [role, setRole]);

  return <RoleContext.Provider value={value}>{children}</RoleContext.Provider>;
}

export function useRole(): RoleContextValue {
  const value = useContext(RoleContext);
  if (!value) {
    throw new Error('useRole requires a RoleProvider.');
  }
  return value;
}
