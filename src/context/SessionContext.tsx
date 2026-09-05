import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { ANONYMOUS_SESSION, fetchSession, logout, type Session } from '../lib/authApi';
import { onUnauthorized } from '../lib/httpClient';
import type { SessionState } from './session';

export type { Session } from '../lib/authApi';
export type { SessionState } from './session';

interface SessionContextValue {
  state: SessionState;
  /** The session when known; anonymous while loading or unavailable. */
  session: Session;
  /** Ask the backend again. Used after sign-in, join, and any 401. */
  refresh: () => Promise<void>;
  /** Replace the cached session with one a route just returned. */
  setSession: (session: Session) => void;
  /** End every session this browser holds. */
  signOut: () => Promise<void>;
}

const SessionContext = createContext<SessionContextValue | null>(null);

/**
 * Who this browser is, as the backend last said.
 *
 * Replaces the development role switcher. The value here is a cache of
 * `/api/auth/session` used for navigation and chrome; it is never a source of
 * permission. A 401 from any request refreshes it, so an expired or revoked
 * session turns into the sign-in screen on the next click rather than into a
 * page of error banners.
 */
export function SessionProvider({
  children,
  initialSession,
}: {
  children: React.ReactNode;
  /** Test seam: skip the initial fetch and start with this session. */
  initialSession?: Session;
}) {
  const [state, setState] = useState<SessionState>(() =>
    initialSession ? { status: 'ready', session: initialSession } : { status: 'loading' },
  );
  const refreshing = useRef<Promise<void> | null>(null);

  const refresh = useCallback(async () => {
    // Coalesce: several 401s from one page's requests mean one refresh.
    if (refreshing.current) {
      return refreshing.current;
    }
    const task = (async () => {
      try {
        const session = await fetchSession();
        setState({ status: 'ready', session });
      } catch (error) {
        setState({
          status: 'unavailable',
          message:
            error instanceof Error && error.message
              ? error.message
              : 'Could not check who is signed in.',
        });
      } finally {
        refreshing.current = null;
      }
    })();
    refreshing.current = task;
    return task;
  }, []);

  useEffect(() => {
    if (!initialSession) {
      void refresh();
    }
  }, [initialSession, refresh]);

  useEffect(() => onUnauthorized(() => void refresh()), [refresh]);

  const setSession = useCallback((session: Session) => {
    setState({ status: 'ready', session });
  }, []);

  const signOut = useCallback(async () => {
    try {
      await logout();
    } finally {
      // Whatever the backend said, this browser is signed out from the
      // application's point of view: the cookies were cleared or never worked.
      setState({ status: 'ready', session: ANONYMOUS_SESSION });
    }
  }, []);

  const value = useMemo<SessionContextValue>(
    () => ({
      state,
      session: state.status === 'ready' ? state.session : ANONYMOUS_SESSION,
      refresh,
      setSession,
      signOut,
    }),
    [state, refresh, setSession, signOut],
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionContextValue {
  const value = useContext(SessionContext);
  if (!value) {
    throw new Error('useSession requires a SessionProvider.');
  }
  return value;
}
