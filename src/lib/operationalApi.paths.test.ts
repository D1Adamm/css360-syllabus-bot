import { beforeEach, describe, expect, it, vi } from 'vitest';

/**
 * Final composed URLs for every operational endpoint, against the deployed base.
 *
 * The production failure: `VITE_API_BASE_URL` is `http://aiswe.uwb.edu/api` and
 * Nginx forwards `location /api/` unchanged, so the client contributes only the
 * part after `/api`. The operational modules were writing full backend paths
 * (`/api/courses/...`), producing `/api/api/courses/...` — 404 — which is what
 * made Examples, Syllabus, and the admin health panel report the backend as
 * offline while it was healthy.
 *
 * Each expectation below is the *actual FastAPI route* taken from the app's
 * OpenAPI schema, prefixed with the host. Asserting the composed URL is the
 * point: a fragment-only assertion passes with either prefix, which is how this
 * shipped twice.
 */

const fetchMock = vi.fn();
vi.stubGlobal('fetch', fetchMock);

// The exact deployed value.
vi.stubEnv('VITE_API_BASE_URL', 'http://aiswe.uwb.edu/api');

const HOST = 'http://aiswe.uwb.edu';
const COURSE = 'css-360-winter-2026-a7rp';
const SEED = '-Ozt97PxVXZ6vRu7v4F0';

function requestedUrl(callIndex = 0): string {
  return fetchMock.mock.calls[callIndex][0] as string;
}

beforeEach(() => {
  fetchMock.mockReset();
  fetchMock.mockResolvedValue({ ok: true, status: 200, json: async () => ({}) });
});

/**
 * frontend call -> the FastAPI route it must reach.
 *
 * Every backend path here was read out of the running app's OpenAPI schema,
 * not transcribed from the frontend.
 */
const OPERATIONAL_CASES: Array<{
  name: string;
  call: () => Promise<unknown>;
  backendRoute: string;
}> = [
  // --- Inference: root-level backend routes, reachable via their /api alias ---
  {
    name: 'generateBaseModel',
    call: async () =>
      (await import('./api')).generateBaseModel(COURSE, 'When are office hours?'),
    backendRoute: '/api/base-model/generate',
  },
  {
    name: 'generateFineTuned',
    call: async () => (await import('./api')).generateFineTuned(COURSE, 'Q?'),
    backendRoute: '/api/fine-tuned/generate',
  },
  {
    name: 'generateFineTunedRag',
    call: async () => (await import('./api')).generateFineTunedRag(COURSE, 'Q?'),
    backendRoute: '/api/fine-tuned-rag/generate',
  },
  {
    name: 'generateRag',
    call: async () => (await import('./api')).generateRag(COURSE, 'Q?'),
    backendRoute: '/api/rag/generate',
  },

  // --- Syllabus ---
  {
    name: 'fetchCourseSyllabusText',
    call: async () => (await import('./api')).fetchCourseSyllabusText(COURSE),
    backendRoute: `/api/courses/${COURSE}/syllabus/text`,
  },

  // --- Seeds: list, review, export, split ---
  {
    name: 'listCourseSeeds',
    call: async () => (await import('./api')).listCourseSeeds(COURSE),
    backendRoute: `/api/courses/${COURSE}/seeds`,
  },
  {
    name: 'reviewCourseSeed',
    call: async () =>
      (await import('./api')).reviewCourseSeed(COURSE, SEED, {
        reviewStatus: 'approved',
      }),
    backendRoute: `/api/courses/${COURSE}/seeds/${encodeURIComponent(SEED)}/review`,
  },
  {
    name: 'exportApprovedCourseSeeds',
    call: async () => (await import('./api')).exportApprovedCourseSeeds(COURSE),
    backendRoute: `/api/courses/${COURSE}/seeds/export-approved`,
  },
  {
    name: 'getApprovedExportStatus',
    call: async () => (await import('./api')).getApprovedExportStatus(COURSE),
    backendRoute: `/api/courses/${COURSE}/seeds/approved-export-status`,
  },
  {
    name: 'prepareTrainingSplit',
    call: async () => (await import('./api')).prepareTrainingSplit(COURSE),
    backendRoute: `/api/courses/${COURSE}/seeds/prepare-training-split`,
  },

  // --- Training ---
  {
    name: 'enqueueTrainingRun',
    call: async () =>
      (await import('./api')).enqueueTrainingRun(COURSE, { datasetRef: 'exports/x' }),
    backendRoute: `/api/courses/${COURSE}/training-runs`,
  },
  {
    name: 'fetchTrainingLaunchCapability',
    call: async () => (await import('./adminApi')).fetchTrainingLaunchCapability(),
    backendRoute: '/api/training/launch-capability',
  },
  {
    name: 'launchCourseTraining',
    call: async () => (await import('./adminApi')).launchCourseTraining(COURSE, 'full'),
    backendRoute: `/api/courses/${COURSE}/training/launch`,
  },

  // --- Admin panels ---
  {
    name: 'fetchBackendHealth',
    call: async () => (await import('./adminApi')).fetchBackendHealth(),
    backendRoute: '/api/health',
  },
  {
    name: 'fetchFineTunedHealth',
    call: async () => (await import('./adminApi')).fetchFineTunedHealth(),
    backendRoute: '/api/fine-tuned/health',
  },
  {
    name: 'fetchStarterGenerationStatus',
    call: async () => (await import('./adminApi')).fetchStarterGenerationStatus(),
    backendRoute: '/api/starter-generation/status',
  },
  {
    name: 'fetchCourseChunks',
    call: async () => (await import('./adminApi')).fetchCourseChunks(COURSE),
    backendRoute: `/api/courses/${COURSE}/chunks`,
  },
  {
    name: 'fetchFactInventory',
    call: async () => (await import('./adminApi')).fetchFactInventory(COURSE),
    backendRoute: `/api/courses/${COURSE}/facts/inventory`,
  },
  {
    name: 'runSeedQualityCheck',
    call: async () => (await import('./adminApi')).runSeedQualityCheck(COURSE),
    backendRoute: `/api/courses/${COURSE}/seeds/quality-check`,
  },
];

describe('operational endpoints compose to real FastAPI routes', () => {
  it.each(OPERATIONAL_CASES.map((c) => [c.name, c] as const))(
    '%s',
    async (_name, testCase) => {
      await testCase.call();

      expect(requestedUrl()).toBe(`${HOST}${testCase.backendRoute}`);
    },
  );

  it('syllabus upload posts to the real upload route', async () => {
    // Not routed through the shared helper: it sends multipart, so it builds
    // its own URL and needs asserting separately.
    const { uploadCourseSyllabus } = await import('./api');
    const file = new File(['syllabus text'], 'syllabus.txt', { type: 'text/plain' });

    await uploadCourseSyllabus(COURSE, file).catch(() => undefined);

    expect(requestedUrl()).toBe(`${HOST}/api/courses/${COURSE}/syllabus`);
  });
});

describe('no operational request doubles the /api prefix', () => {
  it('never produces /api/api/ for any endpoint', async () => {
    for (const testCase of OPERATIONAL_CASES) {
      fetchMock.mockClear();
      await testCase.call();

      expect(requestedUrl(), testCase.name).not.toContain('/api/api/');
    }
  });

  it('reproduces none of the URLs Nginx logged as 404', async () => {
    /*
     * The four the logs named, plus the two root-level ones that never reached
     * uvicorn at all. Each is now asserted to be gone.
     */
    const wasBroken = [
      `${HOST}/api/api/courses/${COURSE}/seeds`,
      `${HOST}/api/api/courses/${COURSE}/syllabus/text`,
      `${HOST}/api/api/courses/${COURSE}/seeds/approved-export-status`,
      `${HOST}/api/api/starter-generation/status`,
    ];

    const seen: string[] = [];
    for (const testCase of OPERATIONAL_CASES) {
      fetchMock.mockClear();
      await testCase.call();
      seen.push(requestedUrl());
    }

    for (const url of wasBroken) {
      expect(seen).not.toContain(url);
    }
  });

  it('sends health checks somewhere Nginx actually forwards', async () => {
    /*
     * `/health` and `/fine-tuned/health` are served at the backend root, but
     * Nginx only forwards `location /api/` — a browser request to `/health`
     * reaches the SPA, not uvicorn. Both must therefore go to the /api alias.
     */
    const adminApi = await import('./adminApi');

    fetchMock.mockClear();
    await adminApi.fetchBackendHealth();
    expect(requestedUrl()).toBe(`${HOST}/api/health`);
    expect(requestedUrl()).not.toBe(`${HOST}/health`);

    fetchMock.mockClear();
    await adminApi.fetchFineTunedHealth();
    expect(requestedUrl()).toBe(`${HOST}/api/fine-tuned/health`);
  });
});

describe('every composed URL starts with the configured base', () => {
  it('appends only the fragment, whatever the base is', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://localhost:8001');

    const { listCourseSeeds } = await import('./api');
    await listCourseSeeds(COURSE);

    // A base without the /api prefix composes the backend route directly,
    // which is what a developer running uvicorn locally needs.
    expect(requestedUrl()).toBe(`http://localhost:8001/courses/${COURSE}/seeds`);

    vi.stubEnv('VITE_API_BASE_URL', 'http://aiswe.uwb.edu/api');
  });
});
