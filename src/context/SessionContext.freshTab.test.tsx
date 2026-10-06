/** @vitest-environment jsdom */
/*
 * Investigation: "a second AISWE tab stays on the loading screen forever".
 *
 * Each test mounts the real application tree (StrictMode, SessionProvider,
 * ComparisonRunProvider, AppRoutes) the way a brand-new browser tab does: no
 * initialSession seam, the real authApi/httpClient, and only `fetch` stubbed 
 * standing in for the backend plus the HttpOnly cookie the browser would send.
 */
import { StrictMode } from 'react';
import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

import { AppRoutes } from '../App';
import { ComparisonRunProvider } from './ComparisonRunContext';
import { SessionProvider } from './SessionContext';

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

type Handler = (path: string) => Promise<Response>;

/** The backend as the authenticated cookie sees it: a professor is signed in. */
const signedInBackend: Handler = async (path) => {
  if (path === '/auth/session') return json(200, PROFESSOR_PAYLOAD);
  if (path === '/db/courses') return json(200, COURSE_LIST);
  return json(404, { detail: 'not part of this test' });
};

function installFetch(handler: Handler) {
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = String(input);
    return handler(url.startsWith(BASE) ? url.slice(BASE.length) : url);
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
    ['network error', async () => { throw new TypeError('Failed to fetch'); }],
  ])('%s → leaves the loader and lands on sign-in', async (_label, sessionResponse) => {
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

  it('REPRODUCTION: a session request that never settles leaves the loader up indefinitely', async () => {
    installFetch((path) =>
      path === '/auth/session' ? new Promise<Response>(() => {}) : signedInBackend(path),
    );
    openTab('/professor/courses');
    expect(screen.getByText(CHECKING)).toBeInTheDocument();

    // Ten minutes of wall clock: nothing in the client ever gives up.
    vi.useFakeTimers();
    await act(async () => {
      vi.advanceTimersByTime(10 * 60 * 1000);
    });
    vi.useRealTimers();

    expect(screen.getByText(CHECKING)).toBeInTheDocument();
    expect(screen.getByTestId('pathname')).toHaveTextContent('/professor/courses');
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
  const acquire = () =>
    new Promise<void>((resolve) => {
      if (active < limit) {
        active += 1;
        resolve();
      } else {
        waiting.push(() => {
          active += 1;
          resolve();
        });
      }
    });
  const release = () => {
    active -= 1;
    waiting.shift()?.();
  };
  return {
    through(handler: Handler): Handler {
      return async (path) => {
        await acquire();
        try {
          return await handler(path);
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
  it('REPRODUCTION: tab 1 holding 6 long requests leaves tab 2 on the loader until one finishes', async () => {
    const pool = createHostPool(6);
    // Tab 1's in-flight generation requests (Compare: 4 per course, no client
    // timeout, serialized server-side behind one Ollama lock).
    const finishTab1: Array<() => void> = [];
    const tab1Request = pool.through(
      () => new Promise<Response>((resolve) => finishTab1.push(() => resolve(json(200, {})))),
    );
    for (let i = 0; i < 6; i += 1) {
      void tab1Request('/rag/generate');
    }
    await waitFor(() => expect(finishTab1).toHaveLength(6));

    // Tab 2: a fresh tab with a perfectly valid cookie.
    installFetch(pool.through(signedInBackend));
    openTab('/professor/courses');

    await waitFor(() => expect(pool.queued).toBeGreaterThan(0));
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.getByText(CHECKING)).toBeInTheDocument();

    // One of tab 1's requests completes: tab 2's session request gets a socket.
    act(() => finishTab1.shift()!());
    await expectProfessorCourses();

    finishTab1.forEach((finish) => finish());
  });
});
