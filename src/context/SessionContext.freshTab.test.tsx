/** @vitest-environment jsdom */
/*
 * Regression: "a second AISWE tab stays on the loading screen forever".
 *
 * Each test mounts the real application tree (StrictMode, SessionProvider,
 * ComparisonRunProvider, AppRoutes) the way a brand-new browser tab does: no
 * initialSession seam, the real authApi/httpClient, and only `fetch` stubbed,
 * standing in for the backend plus the HttpOnly cookie the browser would send.
 *
 * The stub honours the request's AbortSignal the way a browser's fetch does
 * (an aborted request rejects, and one still waiting for a connection leaves
 * the queue). A stub that ignored it could not show the client giving up.
 */
import { StrictMode } from 'react';
import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

import { AppRoutes } from '../App';
import { ComparisonRunProvider } from './ComparisonRunContext';
import { SessionProvider } from './SessionContext';
import { SESSION_CHECK_RETRY_DELAYS_MS, SESSION_CHECK_TIMEOUT_MS } from './session';

const COURSE_ID = 'css-360-winter-2026-a7rp';
const BASE = 'https://aiswe.uwb.edu/api';

const PROFESSOR_PAYLOAD = {
  user: {
    userId: 'u-1',
    email: 'prof@uw.edu',
    displayName: 'Prof',
    role: 'professor',
    courseIds: [COURSE_ID],
  },
  participants: [],
  participant: null,
};

const COURSE_LIST = {
  count: 1,
  courses: [
    {
      courseId: COURSE_ID,
      metadata: {
        name: 'CSS 360',
        title: 'Software Engineering',
        term: 'Winter 2026',
        instructorName: 'Prof',
        createdAt: '2026-01-01T00:00:00Z',
        syllabusStatus: 'indexed',
        syllabusFileName: null,
        syllabusType: null,
        chunkCount: 3,
      },
    },
  ],
};

const CHECKING = 'Checking your session…';

function json(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

type Handler = (path: string, signal?: AbortSignal) => Promise<Response>;

/** Everything the client may spend before it must leave the loader. */
const SESSION_CHECK_BUDGET_MS =
  SESSION_CHECK_TIMEOUT_MS * (SESSION_CHECK_RETRY_DELAYS_MS.length + 1) +
  SESSION_CHECK_RETRY_DELAYS_MS.reduce((total, delay) => total + delay, 0);

function abortError(): DOMException {
  return new DOMException('The operation was aborted.', 'AbortError');
}

/** A request that gets no answer until the browser gives up on it. */
function neverAnswered(signal?: AbortSignal): Promise<Response> {
  return new Promise<Response>((_resolve, reject) => {
    signal?.addEventListener('abort', () => reject(abortError()), { once: true });
  });
}

/** Advance fake time, letting React and every promise settle in between. */
async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

/** The backend as the authenticated cookie sees it: a professor is signed in. */
const signedInBackend: Handler = async (path) => {
  if (path === '/auth/session') return json(200, PROFESSOR_PAYLOAD);
  if (path === '/db/courses') return json(200, COURSE_LIST);
  return json(404, { detail: 'not part of this test' });
};

function installFetch(handler: Handler) {
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const signal = init?.signal ?? undefined;
    if (signal?.aborted) {
      return Promise.reject(abortError());
    }
    return handler(url.startsWith(BASE) ? url.slice(BASE.length) : url, signal);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

function LocationProbe() {
  const { pathname } = useLocation();
  return <span data-testid="pathname">{pathname}</span>;
}

/** A fresh tab: new JS heap, new React tree, whatever sessionStorage it has. */
function openTab(url: string) {
  return render(
    <StrictMode>
      <MemoryRouter initialEntries={[url]}>
        <SessionProvider>
          <ComparisonRunProvider>
            <AppRoutes />
            <LocationProbe />
          </ComparisonRunProvider>
        </SessionProvider>
      </MemoryRouter>
    </StrictMode>,
  );
}

async function expectProfessorCourses() {
  await waitFor(() => expect(screen.getByTestId('pathname')).toHaveTextContent('/professor/courses'));
  await waitFor(() => expect(screen.queryByText(CHECKING)).not.toBeInTheDocument());
  expect(await screen.findByText('CSS 360')).toBeInTheDocument();
}

beforeEach(() => {
  vi.stubEnv('VITE_API_BASE_URL', BASE);
  window.sessionStorage.clear();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
  window.sessionStorage.clear();
});

describe('fresh tab with a valid backend session', () => {
  it.each(['/', '/professor', '/professor/courses'])(
    'empty sessionStorage, direct navigation to %s reaches the professor course list',
    async (url) => {
      const fetchMock = installFetch(signedInBackend);
      openTab(url);
      await expectProfessorCourses();
      // Everything came from /api; nothing from the other tab.
      expect(fetchMock.mock.calls.map(([u]) => String(u))).toContain(`${BASE}/auth/session`);
    },
  );

  it('direct navigation to a course page passes the course-staff guard', async () => {
    installFetch(signedInBackend);
    openTab(`/professor/course/${COURSE_ID}/model`);
    await waitFor(() => expect(screen.queryByText(CHECKING)).not.toBeInTheDocument());
    expect(screen.getByTestId('pathname')).toHaveTextContent(`/professor/course/${COURSE_ID}/model`);
  });

  it('a second tab with copied sessionStorage ("Duplicate tab") behaves the same', async () => {
    window.sessionStorage.setItem(
      `sml.run.${COURSE_ID}`,
      JSON.stringify({
        runId: 'r1',
        courseId: COURSE_ID,
        question: 'q',
        matchedComparisonId: null,
        createdAt: '2026-10-01T00:00:00Z',
        responses: {
          base: { text: 'a', error: null, sources: [] },
          rag: { text: 'a', error: null, sources: [] },
          fineTuned: { text: 'a', error: null, sources: [] },
          fineTunedRag: { text: 'a', error: null, sources: [] },
        },
      }),
    );
    installFetch(signedInBackend);
    openTab('/professor/courses');
    await expectProfessorCourses();
  });

  it('hard refresh (unmount, remount the whole tree) re-bootstraps from /api', async () => {
    const fetchMock = installFetch(signedInBackend);
    const first = openTab('/professor/courses');
    await expectProfessorCourses();
    first.unmount();

    openTab('/professor/courses');
    await expectProfessorCourses();
    const sessionCalls = fetchMock.mock.calls.filter(([u]) => String(u).endsWith('/auth/session'));
    expect(sessionCalls.length).toBeGreaterThanOrEqual(2);
  });

  it('malformed sessionStorage entries do not affect bootstrap', async () => {
    window.sessionStorage.setItem(`sml.run.${COURSE_ID}`, '{not json');
    window.sessionStorage.setItem('role', 'student');
    window.sessionStorage.setItem('selectedCourse', 'garbage');
    installFetch(signedInBackend);
    openTab('/professor/courses');
    await expectProfessorCourses();
  });

  it('sessionStorage that throws on access (blocked storage) does not affect bootstrap', async () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new DOMException('denied', 'SecurityError');
    });
    installFetch(signedInBackend);
    openTab('/professor/courses');
    await expectProfessorCourses();
  });
});

describe('fresh tab when /api/auth/session does not answer normally', () => {
  it.each([
    ['500', async () => json(500, { detail: 'boom' })],
    ['503 with non-JSON body', async () => ({ ok: false, status: 503, json: async () => { throw new Error('html'); } }) as unknown as Response],
    ['401', async () => json(401, { detail: 'Sign in to continue.' })],
  ])('%s → an answer is final: leaves the loader and lands on sign-in', async (_label, sessionResponse) => {
    installFetch((path) => (path === '/auth/session' ? sessionResponse() : signedInBackend(path)));
    openTab('/professor/courses');
    await waitFor(() => expect(screen.getByTestId('pathname')).toHaveTextContent('/login'));
    expect(screen.queryByText(CHECKING)).not.toBeInTheDocument();
  });

  it('malformed 200 body (missing participants) is tolerated', async () => {
    installFetch((path) =>
      path === '/auth/session'
        ? Promise.resolve(json(200, { user: PROFESSOR_PAYLOAD.user }))
        : signedInBackend(path),
    );
    openTab('/professor/courses');
    await expectProfessorCourses();
  });

  it('network error → retried, then leaves the loader and lands on sign-in', async () => {
    vi.useFakeTimers();
    const fetchMock = installFetch((path) =>
      path === '/auth/session' ? Promise.reject(new TypeError('Failed to fetch')) : signedInBackend(path),
    );
    openTab('/professor/courses');
    await advance(SESSION_CHECK_BUDGET_MS);
    vi.useRealTimers();

    await waitFor(() => expect(screen.getByTestId('pathname')).toHaveTextContent('/login'));
    expect(screen.queryByText(CHECKING)).not.toBeInTheDocument();
    const sessionCalls = fetchMock.mock.calls.filter(([u]) => String(u).endsWith('/auth/session'));
    expect(sessionCalls).toHaveLength(SESSION_CHECK_RETRY_DELAYS_MS.length + 1);
  });

  it('a network error once, then an answer → reaches the course list without a detour', async () => {
    vi.useFakeTimers();
    let failures = 1;
    installFetch((path) => {
      if (path === '/auth/session' && failures > 0) {
        failures -= 1;
        return Promise.reject(new TypeError('Failed to fetch'));
      }
      return signedInBackend(path);
    });
    openTab('/professor/courses');
    await advance(SESSION_CHECK_RETRY_DELAYS_MS[0]);
    vi.useRealTimers();

    await expectProfessorCourses();
  });

  it('a session request that never gets an answer is stopped, and the tab leaves the loader', async () => {
    vi.useFakeTimers();
    const signals: AbortSignal[] = [];
    installFetch((path, signal) => {
      if (path !== '/auth/session') {
        return signedInBackend(path);
      }
      if (signal) {
        signals.push(signal);
      }
      return neverAnswered(signal);
    });
    openTab('/professor/courses');
    expect(screen.getByText(CHECKING)).toBeInTheDocument();

    // Just short of the first timeout: still checking, nothing given up yet.
    await advance(SESSION_CHECK_TIMEOUT_MS - 1);
    expect(screen.getByText(CHECKING)).toBeInTheDocument();
    expect(signals.some((signal) => signal.aborted)).toBe(false);

    await advance(SESSION_CHECK_BUDGET_MS);
    vi.useRealTimers();

    // Every attempt carried a signal and every one was stopped: none is left
    // occupying a connection the page will need.
    expect(signals).toHaveLength(SESSION_CHECK_RETRY_DELAYS_MS.length + 1);
    expect(signals.every((signal) => signal.aborted)).toBe(true);
    await waitFor(() => expect(screen.getByTestId('pathname')).toHaveTextContent('/login'));
    expect(screen.queryByText(CHECKING)).not.toBeInTheDocument();
  });
});

/*
 * Model of the browser's HTTP/1.1 socket pool.
 *
 * Production (nginx/1.26.3 on aiswe.uwb.edu) negotiates HTTP/1.1 only, and
 * Chromium/Firefox allow 6 concurrent connections per host per profile, shared
 * by every tab. A request beyond that waits in the browser ("Stalled" in
 * DevTools) without being sent, so neither nginx's proxy_read_timeout nor the
 * backend ever sees it.
 */
function createHostPool(limit: number) {
  let active = 0;
  const waiting: Array<() => void> = [];
  const acquire = (signal?: AbortSignal) =>
    new Promise<void>((resolve, reject) => {
      if (active < limit) {
        active += 1;
        resolve();
        return;
      }
      const grant = () => {
        active += 1;
        resolve();
      };
      waiting.push(grant);
      // A request stopped while waiting for a connection leaves the queue.
      signal?.addEventListener(
        'abort',
        () => {
          const index = waiting.indexOf(grant);
          if (index >= 0) {
            waiting.splice(index, 1);
            reject(abortError());
          }
        },
        { once: true },
      );
    });
  const release = () => {
    active -= 1;
    waiting.shift()?.();
  };
  return {
    through(handler: Handler): Handler {
      return async (path, signal) => {
        await acquire(signal);
        try {
          return await handler(path, signal);
        } finally {
          release();
        }
      };
    },
    get queued() {
      return waiting.length;
    },
  };
}

describe('two tabs sharing one HTTP/1.1 connection pool', () => {
  /** Tab 1: a Compare run's long generation requests, holding `count` connections. */
  function busyFirstTab(pool: ReturnType<typeof createHostPool>, count: number) {
    const finish: Array<() => void> = [];
    const request = pool.through(
      () => new Promise<Response>((resolve) => finish.push(() => resolve(json(200, {})))),
    );
    for (let i = 0; i < count; i += 1) {
      void request('/rag/generate');
    }
    return finish;
  }

  it('tab 2 waits while tab 1 holds every connection, then signs in once one frees up', async () => {
    vi.useFakeTimers();
    const pool = createHostPool(6);
    const finishTab1 = busyFirstTab(pool, 6);
    await advance(0);
    expect(finishTab1).toHaveLength(6);

    installFetch(pool.through(signedInBackend));
    openTab('/professor/courses');
    await advance(0);
    expect(pool.queued).toBe(1);
    expect(screen.getByText(CHECKING)).toBeInTheDocument();

    // The first check is stopped and leaves the browser's queue.
    await advance(SESSION_CHECK_TIMEOUT_MS);
    expect(pool.queued).toBe(0);
    expect(screen.getByText(CHECKING)).toBeInTheDocument();

    // One of tab 1's generations finishes; the retry gets that connection.
    await act(async () => finishTab1.shift()!());
    await advance(SESSION_CHECK_RETRY_DELAYS_MS[0]);
    vi.useRealTimers();

    await expectProfessorCourses();
    finishTab1.forEach((finish) => finish());
  });

  it('tab 2 never waits indefinitely, even if tab 1 never lets a connection go', async () => {
    vi.useFakeTimers();
    const pool = createHostPool(6);
    const finishTab1 = busyFirstTab(pool, 6);
    installFetch(pool.through(signedInBackend));
    openTab('/professor/courses');

    await advance(SESSION_CHECK_BUDGET_MS);
    vi.useRealTimers();

    await waitFor(() => expect(screen.queryByText(CHECKING)).not.toBeInTheDocument());
    expect(screen.getByTestId('pathname')).toHaveTextContent('/login');
    expect(pool.queued).toBe(0);
    finishTab1.forEach((finish) => finish());
  });
});
