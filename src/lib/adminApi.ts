import { getApiBaseUrl, requestJson } from './httpClient';

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


export { ApiError } from './httpClient';

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

/** What the shared inference service reports about itself, as a browser may read it.
 *
 * No hostname, port or service URL. `/api/fine-tuned/health` needs no
 * credential, and those three describe how to reach a Tillicum compute node
 * rather than what the service is doing — the same rule `ServingSession`
 * already follows. The backend drops them in
 * `finetuned_client.public_service_health`.
 */
export interface FineTunedHealth {
  status: string;
  model?: string | null;
  adapterLoaded?: boolean | null;
  /** Courses the running service can answer for, and with which versions. */
  courses?: FineTunedHealthCourse[];
  /** Wall clock left in the serving allocation, when the service reports one. */
  secondsRemaining?: number | null;
}

export interface FineTunedHealthCourse {
  courseId?: string;
  versions?: string[];
  currentVersion?: string;
}

export interface StarterGenerationStatus {
  active: boolean;
  courseId?: string | null;
  operation?: string | null;
  startedAt?: string | null;
}

function getJson<T>(path: string): Promise<T> {
  return requestJson<T>(path, {
    method: 'GET',
    fallbackErrorMessage: 'The request failed.',
  });
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

/**
 * How long a diagnostic may take before the page gives up on it.
 *
 * Every probe on the admin course page either reads a file, computes over
 * stored rows, or asks the backend for the *status* of a long build rather
 * than waiting on the build itself. None of them should take a minute; a
 * request that does is a stuck proxy or a stuck backend, and the control must
 * come back with a message rather than spin until someone reloads.
 */
export const DIAGNOSTIC_TIMEOUT_MS = 30_000;

function postJsonAdmin<T>(path: string, body: unknown, timeoutMs?: number): Promise<T> {
  return requestJson<T>(path, {
    method: 'POST',
    json: body,
    fallbackErrorMessage: 'The request failed.',
    ...(timeoutMs ? { timeoutMs } : {}),
  });
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
  return requestJson<CourseChunksResponse>(`/courses/${courseId}/chunks`, {
    method: 'GET',
    fallbackErrorMessage: 'The request failed.',
    timeoutMs: DIAGNOSTIC_TIMEOUT_MS,
  });
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

/** The backend's 202 body while an inventory is being extracted. */
export interface FactInventoryBuildingResponse {
  courseId: string;
  status: 'building';
  startedAt?: string | null;
}

export type FactInventoryProbeResult =
  | { status: 'ready'; inventory: FactInventoryResponse }
  | { status: 'building'; startedAt: string | null };

function isBuilding(
  body: FactInventoryResponse | FactInventoryBuildingResponse,
): body is FactInventoryBuildingResponse {
  return (body as FactInventoryBuildingResponse).status === 'building';
}

/**
 * Extraction-only. The backend docstring is explicit that this does NOT
 * generate seeds; it builds or reuses the inspectable fact inventory.
 *
 * Asked with `wait: false`, so the backend answers at once: the cached
 * inventory when it has one, otherwise `building` after starting (or joining)
 * the extraction in the background. A rebuild is every batch of the syllabus
 * through the local model on the CPU, and the page used to hold the request
 * open for the whole of it — which is how "Building…" came to mean "until
 * somebody reloads". Call again to collect the result; a failed build comes
 * back as an `ApiError` on the poll after it fails, and the next call starts
 * a fresh one.
 */
export async function requestFactInventory(
  courseId: string,
): Promise<FactInventoryProbeResult> {
  const body = await postJsonAdmin<FactInventoryResponse | FactInventoryBuildingResponse>(
    `/courses/${courseId}/facts/inventory`,
    { wait: false },
    DIAGNOSTIC_TIMEOUT_MS,
  );
  if (isBuilding(body)) {
    return { status: 'building', startedAt: body.startedAt ?? null };
  }
  return { status: 'ready', inventory: body };
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
    DIAGNOSTIC_TIMEOUT_MS,
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
