import { ApiError, getApiBaseUrl } from './api';

/**
 * Request paths are RELATIVE to `VITE_API_BASE_URL`, which carries the `/api`
 * prefix in deployment (`http://aiswe.uwb.edu/api`).
 *
 * So a backend route of `/api/courses/{id}/seeds` is written here as
 * `/courses/{id}/seeds`. Writing the full backend path produced
 * `…/api/api/courses/…`, which Nginx forwards unchanged and FastAPI has no
 * route for — that 404 is what made Examples, Syllabus, and the admin panels
 * report the backend as unavailable.
 *
 * Routes the backend also serves at the root (`/health`, `/rag/generate`) are
 * written here without a prefix too. They compose to `/api/health`,
 * `/api/rag/generate`, which the backend now serves as aliases — Nginx only
 * forwards `location /api/`, so the root paths never reach it from a browser.
 */


export { ApiError } from './api';

/**
 * Admin-only API clients.
 *
 * These call endpoints the backend has always exposed but the UI never used.
 * They are read-only diagnostics; nothing here changes a request or response
 * shape, and nothing here is reachable from a student or professor surface.
 */

export interface BackendHealth {
  status: string;
  service: string;
}

export interface FineTunedHealth {
  status: string;
  model?: string | null;
  adapterLoaded?: boolean | null;
  hostname?: string | null;
  port?: number | null;
  serviceUrl?: string | null;
}

export interface StarterGenerationStatus {
  active: boolean;
  courseId?: string | null;
  operation?: string | null;
  startedAt?: string | null;
}

async function getJson<T>(path: string): Promise<T> {
  const baseUrl = getApiBaseUrl();

  if (!baseUrl) {
    throw new ApiError('The service is not configured.');
  }

  let response: Response;

  try {
    response = await fetch(`${baseUrl}${path}`, { method: 'GET' });
  } catch {
    throw new ApiError('The service could not be reached.');
  }

  if (!response.ok) {
    throw new ApiError(`Request failed with status ${response.status}.`, response.status);
  }

  return (await response.json()) as T;
}

export function fetchBackendHealth(): Promise<BackendHealth> {
  return getJson<BackendHealth>('/health');
}

export function fetchFineTunedHealth(): Promise<FineTunedHealth> {
  return getJson<FineTunedHealth>('/fine-tuned/health');
}

export function fetchStarterGenerationStatus(): Promise<StarterGenerationStatus> {
  return getJson<StarterGenerationStatus>('/starter-generation/status');
}

/** The configured backend origin, shown only in admin diagnostics. */
export function getConfiguredApiBaseUrl(): string | null {
  return getApiBaseUrl();
}

/* ---------------------------------------------------------------------------
 * Per-course diagnostics.
 *
 * These endpoints have existed on the backend since before this redesign and
 * were only ever reachable with curl. They are read-only inspections of what
 * the pipeline produced, which is exactly what an admin needs and exactly what
 * a professor should never see.
 * ------------------------------------------------------------------------- */

async function postJsonAdmin<T>(path: string, body: unknown): Promise<T> {
  const baseUrl = getApiBaseUrl();

  if (!baseUrl) {
    throw new ApiError('The service is not configured.');
  }

  let response: Response;

  try {
    response = await fetch(`${baseUrl}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError('The service could not be reached.');
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}.`;
    try {
      const errorBody = (await response.json()) as { detail?: string };
      if (typeof errorBody.detail === 'string' && errorBody.detail.trim() !== '') {
        detail = errorBody.detail;
      }
    } catch {
      // Keep the status-based message when the body is not JSON.
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

export interface CourseChunk {
  chunkId: string;
  sectionTitle: string;
  text: string;
  order: number;
}

export interface CourseChunksResponse {
  courseId: string;
  chunkCount: number;
  chunks: CourseChunk[];
  indexVersion?: number | null;
  documentTitle?: string | null;
}

export function fetchCourseChunks(courseId: string): Promise<CourseChunksResponse> {
  return getJson<CourseChunksResponse>(`/courses/${courseId}/chunks`);
}

export interface FactInventoryResponse {
  courseId: string;
  model: string;
  factCount: number;
  droppedCount?: number;
  duplicatesRemoved?: number;
  fallbackUsed?: boolean;
  cached?: boolean;
  countsByScope?: Record<string, number>;
  countsByKind?: Record<string, number>;
}

/**
 * Extraction-only. The backend docstring is explicit that this does NOT
 * generate seeds; it builds or reuses the inspectable fact inventory.
 */
export function fetchFactInventory(courseId: string): Promise<FactInventoryResponse> {
  return postJsonAdmin<FactInventoryResponse>(
    `/courses/${courseId}/facts/inventory`,
    {},
  );
}

export interface SeedQualityCheckResponse {
  courseId: string;
  report: Record<string, unknown>;
}

export function runSeedQualityCheck(
  courseId: string,
): Promise<SeedQualityCheckResponse> {
  return postJsonAdmin<SeedQualityCheckResponse>(
    `/courses/${courseId}/seeds/quality-check`,
    {},
  );
}

/* ---------------------------------------------------------------------------
 * Training launch (deprecated)
 *
 * The browser never runs ssh, rsync, or sbatch. It asked the backend, which
 * owns that boundary and shells out to the existing sync and launcher scripts.
 *
 * No page calls these now. Submitting from a web request needed a
 * non-interactive session to a cluster that only offers interactive logins, so
 * the endpoint stayed disabled behind TRAINING_LAUNCH_ENABLED. Administrators
 * enqueue a training run instead (see `queueTraining.ts`), and the run is
 * claimed on the cluster by someone already logged in.
 * ------------------------------------------------------------------------- */

export interface TrainingLaunchCapability {
  enabled: boolean;
  reason: string;
}

export function fetchTrainingLaunchCapability(): Promise<TrainingLaunchCapability> {
  return getJson<TrainingLaunchCapability>('/training/launch-capability');
}

export interface TrainingLaunchResponse {
  courseId: string;
  jobId: string;
  mode: string;
  submittedAt: string;
  trainCount: number;
  validationCount: number;
}

export function launchCourseTraining(
  courseId: string,
  mode: 'smoke' | 'full' = 'full',
): Promise<TrainingLaunchResponse> {
  return postJsonAdmin<TrainingLaunchResponse>(
    `/courses/${courseId}/training/launch`,
    { mode },
  );
}
