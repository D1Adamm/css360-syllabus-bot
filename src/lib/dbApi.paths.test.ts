import { beforeEach, describe, expect, it, vi } from 'vitest';

/**
 * Request URLs built against the deployed API base.
 *
 * The production bug this pins: `VITE_API_BASE_URL` on the VM is
 * `http://aiswe.uwb.edu/api`, and `dbApi` was also writing `/api/db/...`. Every
 * persistence request therefore went to `/api/api/db/...` and Nginx returned
 * 404 — the whole application read nothing, while the backend was healthy and
 * its routes were correct.
 *
 * Asserting the composed URL is the only way to catch this. A test that checks
 * the path fragment alone passes with either prefix, which is exactly how it
 * shipped.
 */

const fetchMock = vi.fn();
vi.stubGlobal('fetch', fetchMock);

// The real deployed value, not a placeholder.
vi.stubEnv('VITE_API_BASE_URL', 'http://aiswe.uwb.edu/api');

const COURSE = 'css-360-winter-2026-a7rp';
const SEED = '-Ozt97PxVXZ6vRu7v4F0';
const RUN = 'run-20260812t100000z-abc123';

function requestedUrl(callIndex = 0): string {
  return fetchMock.mock.calls[callIndex][0] as string;
}

beforeEach(() => {
  fetchMock.mockReset();
  fetchMock.mockResolvedValue({ ok: true, status: 200, json: async () => ({}) });
});

describe('the reported production URL', () => {
  it('builds http://aiswe.uwb.edu/api/db/courses', async () => {
    const { listCourses } = await import('./dbApi');
    await listCourses();

    expect(requestedUrl()).toBe('http://aiswe.uwb.edu/api/db/courses');
  });

  it('never doubles the /api prefix', async () => {
    const { listCourses } = await import('./dbApi');
    await listCourses();

    expect(requestedUrl()).not.toBe('http://aiswe.uwb.edu/api/api/db/courses');
    expect(requestedUrl()).not.toContain('/api/api/');
  });
});

describe('every dbApi endpoint, composed against the deployed base', () => {
  /**
   * All 24 operations, not just the one that was reported.
   *
   * Seventeen of them are built from a shared `coursePath`, so one fix covered
   * them — but "covered by the same helper" is a property worth asserting
   * rather than assuming, and the two course-collection paths do not use it.
   */
  const cases: Array<[string, (api: typeof import('./dbApi')) => Promise<unknown>, string]> =
    [
      ['listCourses', (api) => api.listCourses(), '/api/db/courses'],
      ['getCourse', (api) => api.getCourse(COURSE), `/api/db/courses/${COURSE}`],
      [
        'createCourse',
        (api) =>
          api.createCourse({
            courseId: COURSE,
            name: 'CSS 360',
            title: 'T',
            term: 'Winter 2026',
            instructorName: 'I',
            createdAt: '2026-01-01T00:00:00.000Z',
            syllabusStatus: 'none',
            syllabusFileName: null,
            syllabusType: null,
            chunkCount: 0,
          }),
        '/api/db/courses',
      ],
      [
        'updateCourse',
        (api) => api.updateCourse(COURSE, { chunkCount: 1 }),
        `/api/db/courses/${COURSE}`,
      ],
      [
        'getStarterSeedGeneration',
        (api) => api.getStarterSeedGeneration(COURSE),
        `/api/db/courses/${COURSE}/starter-seed-generation`,
      ],
      ['listSeeds', (api) => api.listSeeds(COURSE), `/api/db/courses/${COURSE}/seeds`],
      [
        'createSeed',
        (api) => api.createSeed(COURSE, {}),
        `/api/db/courses/${COURSE}/seeds`,
      ],
      [
        'updateSeed',
        (api) => api.updateSeed(COURSE, SEED, {}),
        `/api/db/courses/${COURSE}/seeds/${encodeURIComponent(SEED)}`,
      ],
      [
        'deleteSeed',
        (api) => api.deleteSeed(COURSE, SEED),
        `/api/db/courses/${COURSE}/seeds/${encodeURIComponent(SEED)}`,
      ],
      [
        'listEvaluations',
        (api) => api.listEvaluations(COURSE),
        `/api/db/courses/${COURSE}/evaluations`,
      ],
      [
        'createEvaluation',
        (api) =>
          api.createEvaluation(COURSE, {
            comparisonId: 'c',
            mostAccurate: 'rag',
            mostHelpful: 'rag',
            mostConcise: 'rag',
            bestGrounded: 'rag',
            preferredModel: 'rag',
            hallucinationFlags: [],
            createdAt: '2026-01-01T00:00:00.000Z',
          }),
        `/api/db/courses/${COURSE}/evaluations`,
      ],
      [
        'deleteEvaluation',
        (api) => api.deleteEvaluation(COURSE, 'eval-1'),
        `/api/db/courses/${COURSE}/evaluations/eval-1`,
      ],
      [
        'deleteAllEvaluations',
        (api) => api.deleteAllEvaluations(COURSE),
        `/api/db/courses/${COURSE}/evaluations`,
      ],
      [
        'getCourseModel',
        (api) => api.getCourseModel(COURSE),
        `/api/db/courses/${COURSE}/model`,
      ],
      [
        'getModelRequest',
        (api) => api.getModelRequest(COURSE),
        `/api/db/courses/${COURSE}/model-request`,
      ],
      [
        'createModelRequest',
        (api) => api.createModelRequest(COURSE, 54),
        `/api/db/courses/${COURSE}/model-request`,
      ],
      [
        'updateModelRequest',
        (api) => api.updateModelRequest(COURSE, { status: 'ready' }),
        `/api/db/courses/${COURSE}/model-request`,
      ],
      [
        'listTrainingRuns',
        (api) => api.listTrainingRuns(COURSE),
        `/api/db/courses/${COURSE}/training-runs`,
      ],
      [
        'getTrainingRun',
        (api) => api.getTrainingRun(COURSE, RUN),
        `/api/db/courses/${COURSE}/training-runs/${RUN}`,
      ],
      [
        'createTrainingRun',
        (api) => api.createTrainingRun(COURSE, { mode: 'full', datasetRef: 'x' }),
        `/api/db/courses/${COURSE}/training-runs`,
      ],
      [
        'updateTrainingRun',
        (api) => api.updateTrainingRun(COURSE, RUN, { state: 'queued' }),
        `/api/db/courses/${COURSE}/training-runs/${RUN}`,
      ],
    ];

  it.each(cases)('%s hits the backend route exactly once', async (_name, call, path) => {
    const api = await import('./dbApi');
    await call(api);

    expect(requestedUrl()).toBe(`http://aiswe.uwb.edu${path}`);
  });

  it('no endpoint doubles the prefix', async () => {
    const api = await import('./dbApi');

    for (const [, call] of cases) {
      fetchMock.mockClear();
      await call(api);
      expect(requestedUrl()).not.toContain('/api/api/');
    }
  });
});

describe('a base URL without the /api prefix still composes correctly', () => {
  /**
   * The client contributes `/db/...` and nothing more, so it works against a
   * base that points at the backend root as well as one that already includes
   * `/api`. Which of those is deployed is an Nginx decision, not this file's.
   */
  it('appends the client path to whatever base is configured', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://localhost:8001/api');

    const { listCourses } = await import('./dbApi');
    await listCourses();

    expect(requestedUrl()).toBe('http://localhost:8001/api/db/courses');

    vi.stubEnv('VITE_API_BASE_URL', 'http://aiswe.uwb.edu/api');
  });

  it('drops a trailing slash on the base rather than doubling it', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://aiswe.uwb.edu/api/');

    const { listCourses } = await import('./dbApi');
    await listCourses();

    expect(requestedUrl()).toBe('http://aiswe.uwb.edu/api/db/courses');

    vi.stubEnv('VITE_API_BASE_URL', 'http://aiswe.uwb.edu/api');
  });
});
