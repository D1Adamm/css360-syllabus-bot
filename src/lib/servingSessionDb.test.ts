import { beforeEach, describe, expect, it, vi } from 'vitest';

/**
 * Reading whether a fine-tuned service is up.
 *
 * The question the application could not previously express.
 * `CourseModelVersion.deployment` describes one course's artifact and is
 * durable; a session describes one Slurm allocation with a wall clock on it and
 * belongs to no course in particular, because one allocation serves every course
 * whose adapter it can load.
 *
 * The parser is deliberately conservative in one direction: an unreadable or
 * ambiguous record reads as "not live". Showing a dead session as live is the
 * worse of the two mistakes — it is the one that has somebody demonstrate to a
 * class against a GPU that was released twenty minutes ago.
 */

const getServingSession = vi.fn();

vi.mock('./dbApi', () => ({
  getServingSession: () => getServingSession(),
}));

const { fetchCurrentServingSession, parseServingSession } = await import(
  './servingSessionDb'
);

const SESSION = {
  sessionId: 'serve-264787',
  jobId: '264787',
  state: 'ready',
  startedAt: '2026-08-27T12:00:00.000Z',
  expiresAt: '2026-08-27T14:00:00.000Z',
  updatedAt: '2026-08-27T12:05:00.000Z',
  live: true,
  courses: [{ courseId: 'css-350-spring-2026-n3h9', currentVersion: 'v1' }],
  baseModel: 'meta-llama/Llama-3.2-3B-Instruct',
};

beforeEach(() => {
  getServingSession.mockReset();
});

describe('parseServingSession', () => {
  it('reads a live session with what it is serving', () => {
    const session = parseServingSession(SESSION);

    expect(session).toEqual({
      sessionId: 'serve-264787',
      jobId: '264787',
      state: 'ready',
      startedAt: '2026-08-27T12:00:00.000Z',
      expiresAt: '2026-08-27T14:00:00.000Z',
      updatedAt: '2026-08-27T12:05:00.000Z',
      live: true,
      courses: [{ courseId: 'css-350-spring-2026-n3h9', currentVersion: 'v1' }],
      baseModel: 'meta-llama/Llama-3.2-3B-Instruct',
    });
  });

  it('reads an expired session as not live', () => {
    // The backend decides expiry at read time from the allocation's wall clock,
    // so nothing had to write this state for it to be true.
    const session = parseServingSession({
      ...SESSION,
      state: 'expired',
      live: false,
    });

    expect(session?.state).toBe('expired');
    expect(session?.live).toBe(false);
  });

  it('treats a missing live flag as not live', () => {
    const { live: _live, ...withoutLive } = SESSION;

    expect(parseServingSession(withoutLive)?.live).toBe(false);
  });

  it('treats a non-boolean live flag as not live', () => {
    expect(parseServingSession({ ...SESSION, live: 'yes' })?.live).toBe(false);
  });

  it('rejects a record with no session id', () => {
    const { sessionId: _sessionId, ...withoutId } = SESSION;

    expect(parseServingSession(withoutId)).toBeNull();
  });

  it('rejects an unrecognised state rather than guessing', () => {
    expect(parseServingSession({ ...SESSION, state: 'warming-up' })).toBeNull();
  });

  it('rejects a null or non-object payload', () => {
    expect(parseServingSession(null)).toBeNull();
    expect(parseServingSession('serve-264787')).toBeNull();
    expect(parseServingSession(undefined)).toBeNull();
  });

  it('drops malformed course entries without dropping the session', () => {
    const session = parseServingSession({
      ...SESSION,
      courses: [
        { courseId: 'css-350-spring-2026-n3h9', currentVersion: 'v1' },
        { courseId: 42 },
        null,
      ],
    });

    expect(session?.courses).toEqual([
      { courseId: 'css-350-spring-2026-n3h9', currentVersion: 'v1' },
    ]);
  });

  it('omits courses entirely when none survive parsing', () => {
    const session = parseServingSession({ ...SESSION, courses: [{ bad: true }] });

    expect(session).not.toHaveProperty('courses');
  });

  it('carries no compute hostname or port, because the backend sends none', () => {
    /*
     * Every /api/db route is reachable without a credential, so the node and
     * port are stripped server-side. This asserts the client shape matches: a
     * field that is never sent must not be one anything here can read.
     */
    const session = parseServingSession({ ...SESSION, node: 'g014', port: 8001 });

    expect(session).not.toHaveProperty('node');
    expect(session).not.toHaveProperty('port');
  });
});

describe('fetchCurrentServingSession', () => {
  it('returns the parsed session', async () => {
    getServingSession.mockResolvedValue({ session: SESSION });

    const session = await fetchCurrentServingSession();

    expect(session?.sessionId).toBe('serve-264787');
  });

  it('returns null when nothing is serving', async () => {
    // The resting state of a research GPU allocation, not an error.
    getServingSession.mockResolvedValue({ session: null });

    expect(await fetchCurrentServingSession()).toBeNull();
  });
});
