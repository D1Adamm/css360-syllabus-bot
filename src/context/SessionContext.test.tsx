/** @vitest-environment jsdom */
import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

const fetchSessionMock = vi.hoisted(() => vi.fn());
const logoutMock = vi.hoisted(() => vi.fn());

vi.mock('../lib/authApi', async () => {
  const actual = await vi.importActual<typeof import('../lib/authApi')>('../lib/authApi');
  return { ...actual, fetchSession: fetchSessionMock, logout: logoutMock };
});

import { SessionProvider, useSession } from './SessionContext';
import { getJson } from '../lib/httpClient';

const PROFESSOR = {
  user: {
    userId: 'u-1',
    email: 'prof@uw.edu',
    displayName: 'Prof',
    role: 'professor' as const,
    courseIds: ['css-360-winter-2026-a7rp'],
  },
  participant: null,
};

function Probe() {
  const { state, session, signOut } = useSession();
  return (
    <div>
      <span data-testid="status">{state.status}</span>
      <span data-testid="who">
        {session.user ? session.user.email : session.participant ? 'participant' : 'nobody'}
      </span>
      <button type="button" onClick={() => void signOut()}>
        Sign out
      </button>
    </div>
  );
}

beforeEach(() => {
  fetchSessionMock.mockReset();
  logoutMock.mockReset();
  logoutMock.mockResolvedValue(undefined);
});

afterEach(() => {
  cleanup();
});

describe('SessionProvider', () => {
  it('asks the backend who is signed in on mount', async () => {
    fetchSessionMock.mockResolvedValue(PROFESSOR);
    render(
      <SessionProvider>
        <Probe />
      </SessionProvider>,
    );

    expect(screen.getByTestId('status')).toHaveTextContent('loading');
    await waitFor(() => expect(screen.getByTestId('who')).toHaveTextContent('prof@uw.edu'));
    expect(fetchSessionMock).toHaveBeenCalledTimes(1);
  });

  it('starts from a supplied session without asking', () => {
    render(
      <SessionProvider initialSession={PROFESSOR}>
        <Probe />
      </SessionProvider>,
    );
    expect(screen.getByTestId('status')).toHaveTextContent('ready');
    expect(fetchSessionMock).not.toHaveBeenCalled();
  });

  it('reports the session as unavailable, not anonymous, when the check fails', async () => {
    fetchSessionMock.mockRejectedValue(new Error('down'));
    render(
      <SessionProvider>
        <Probe />
      </SessionProvider>,
    );
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('unavailable'));
    expect(screen.getByTestId('who')).toHaveTextContent('nobody');
  });

  it('re-checks the session when any request is refused with 401', async () => {
    fetchSessionMock.mockResolvedValue({ user: null, participant: null });
    render(
      <SessionProvider initialSession={PROFESSOR}>
        <Probe />
      </SessionProvider>,
    );
    expect(screen.getByTestId('who')).toHaveTextContent('prof@uw.edu');

    // A real refusal through the real client: the provider listens to it.
    vi.stubEnv('VITE_API_BASE_URL', 'https://aiswe.uwb.edu/api');
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: false, status: 401, json: async () => ({ detail: 'x' }) }),
    );
    await act(async () => {
      await getJson('/db/courses', 'refused').catch(() => undefined);
    });
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();

    await waitFor(() => expect(screen.getByTestId('who')).toHaveTextContent('nobody'));
    expect(fetchSessionMock).toHaveBeenCalledTimes(1);
  });

  it('signing out ends the session locally even if the request fails', async () => {
    logoutMock.mockRejectedValue(new Error('offline'));
    render(
      <SessionProvider initialSession={PROFESSOR}>
        <Probe />
      </SessionProvider>,
    );
    await act(async () => {
      screen.getByRole('button', { name: 'Sign out' }).click();
    });
    await waitFor(() => expect(screen.getByTestId('who')).toHaveTextContent('nobody'));
    expect(logoutMock).toHaveBeenCalledTimes(1);
  });
});
