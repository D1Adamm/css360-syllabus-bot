import type {
  CourseMetadata,
  CourseModelRegistry,
  CourseModelRequest,
  EvaluationRecord,
  SeedExample,
  StoredStarterSeedGeneration,
  TrainingRun,
} from '../types';
import { ApiError, getApiBaseUrl } from './api';

/**
 * Typed client for the PostgreSQL-backed `/api/db` backend routes.
 *
 * One place, not a fetch per component. It reuses `getApiBaseUrl` and the
 * `ApiError` shape from `api.ts` so failures surface exactly as they already do
 * everywhere else — the existing error banners keep working without knowing the
 * store changed underneath them.
 *
 * Kept separate from `api.ts` on purpose. That module is the operational
 * surface: generation, RAG, inference, export, training. This one is
 * persistence. During the cutover it has to be obvious at a glance which is
 * which, and after it the two still answer different questions.
 *
 * Every id is URL-encoded. Course ids are validated before they get here, but
 * seed and run ids come from stored records — an imported push id begins with a
 * hyphen and can contain characters that must not be read as path syntax.
 */

const UNREACHABLE = 'The service could not be reached.';

async function request<T>(
  path: string,
  init: RequestInit,
  fallbackErrorMessage: string,
): Promise<T> {
  const baseUrl = getApiBaseUrl();

  if (!baseUrl) {
    throw new ApiError('The service is not configured.');
  }

  let response: Response;

  try {
    response = await fetch(`${baseUrl}${path}`, init);
  } catch {
    throw new ApiError(UNREACHABLE);
  }

  if (!response.ok) {
    let detail = fallbackErrorMessage;

    try {
      const errorBody = (await response.json()) as { detail?: string };
      if (typeof errorBody.detail === 'string' && errorBody.detail.trim() !== '') {
        detail = errorBody.detail;
      }
    } catch {
      // Keep the default message when the error body is not JSON.
    }

    throw new ApiError(detail, response.status);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

function get<T>(path: string, fallbackErrorMessage: string): Promise<T> {
  return request<T>(path, { method: 'GET' }, fallbackErrorMessage);
}

function send<T>(
  method: 'POST' | 'PATCH' | 'DELETE',
  path: string,
  body: unknown,
  fallbackErrorMessage: string,
): Promise<T> {
  return request<T>(
    path,
    {
      method,
      ...(body === undefined
        ? {}
        : {
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
          }),
    },
    fallbackErrorMessage,
  );
}

/**
 * Paths are relative to `VITE_API_BASE_URL`, which already carries the `/api`
 * prefix in deployment (`http://aiswe.uwb.edu/api`).
 *
 * So these start at `/db`, not `/api/db`. Writing the full backend path here
 * produced `…/api/api/db/courses` against the VM and every persistence request
 * 404'd. The backend routes are unchanged and still mounted at `/api/db`; it is
 * only the half the client contributes that belongs below.
 *
 * Every course-scoped endpoint is built from `coursePath`, so this is the one
 * place the prefix is decided.
 */
const DB_ROOT = '/db';

function coursePath(courseId: string): string {
  return `${DB_ROOT}/courses/${encodeURIComponent(courseId)}`;
}

/* ------------------------------------------------------------------------ *
 * Courses
 * ------------------------------------------------------------------------ */

export interface DbCourseRecord {
  courseId: string;
  metadata: CourseMetadata;
}

export interface DbCourseListResponse {
  count: number;
  courses: DbCourseRecord[];
}

export function listCourses(): Promise<DbCourseListResponse> {
  return get<DbCourseListResponse>(
    `${DB_ROOT}/courses`,
    'The backend could not load the course list.',
  );
}

export function getCourse(courseId: string): Promise<DbCourseRecord> {
  return get<DbCourseRecord>(
    coursePath(courseId),
    'The backend could not load this course.',
  );
}

export interface CreateCourseBody extends Omit<CourseMetadata, 'starterSeedGeneration'> {
  courseId: string;
}

export function createCourse(body: CreateCourseBody): Promise<DbCourseRecord> {
  return send<DbCourseRecord>(
    'POST',
    `${DB_ROOT}/courses`,
    body,
    'The backend could not create the course.',
  );
}

export function updateCourse(
  courseId: string,
  patch: Partial<CourseMetadata>,
): Promise<DbCourseRecord> {
  return send<DbCourseRecord>(
    'PATCH',
    coursePath(courseId),
    patch,
    'The backend could not update this course.',
  );
}

/* ------------------------------------------------------------------------ *
 * Starter seed generation status
 * ------------------------------------------------------------------------ */

export interface DbStarterStatusResponse {
  courseId: string;
  starterSeedGeneration: StoredStarterSeedGeneration | null;
}

export function getStarterSeedGeneration(
  courseId: string,
): Promise<DbStarterStatusResponse> {
  return get<DbStarterStatusResponse>(
    `${coursePath(courseId)}/starter-seed-generation`,
    'The backend could not load starter generation status.',
  );
}

/* ------------------------------------------------------------------------ *
 * Seeds
 * ------------------------------------------------------------------------ */

/**
 * A stored seed as the API returns it.
 *
 * Carries both name pairs — instruction/question and response/answer — because
 * the backend emits both and different parts of the UI read different ones.
 */
export interface DbSeedRecord extends SeedExample {
  courseId: string;
  question?: string;
  answer?: string;
}

export interface DbSeedListResponse {
  courseId: string;
  count: number;
  seeds: DbSeedRecord[];
  reviewStatusCounts: Record<string, number>;
}

export interface DbSeedResponse {
  courseId: string;
  seedId: string;
  seed: DbSeedRecord;
}

export function listSeeds(courseId: string): Promise<DbSeedListResponse> {
  return get<DbSeedListResponse>(
    `${coursePath(courseId)}/seeds`,
    'The backend could not load examples for this course.',
  );
}

export function createSeed(
  courseId: string,
  seed: Partial<DbSeedRecord>,
): Promise<DbSeedResponse> {
  return send<DbSeedResponse>(
    'POST',
    `${coursePath(courseId)}/seeds`,
    seed,
    'The backend could not save the example.',
  );
}

export function updateSeed(
  courseId: string,
  seedId: string,
  patch: Partial<DbSeedRecord>,
): Promise<DbSeedResponse> {
  return send<DbSeedResponse>(
    'PATCH',
    `${coursePath(courseId)}/seeds/${encodeURIComponent(seedId)}`,
    patch,
    'The backend could not update the example.',
  );
}

export function deleteSeed(
  courseId: string,
  seedId: string,
): Promise<{ courseId: string; deleted: number }> {
  return send<{ courseId: string; deleted: number }>(
    'DELETE',
    `${coursePath(courseId)}/seeds/${encodeURIComponent(seedId)}`,
    undefined,
    'The backend could not delete the example.',
  );
}

/* ------------------------------------------------------------------------ *
 * Evaluations
 * ------------------------------------------------------------------------ */

export interface DbEvaluationListResponse {
  courseId: string;
  count: number;
  evaluations: EvaluationRecord[];
}

export function listEvaluations(
  courseId: string,
): Promise<DbEvaluationListResponse> {
  return get<DbEvaluationListResponse>(
    `${coursePath(courseId)}/evaluations`,
    'The backend could not load evaluations for this course.',
  );
}

export function createEvaluation(
  courseId: string,
  evaluation: Omit<EvaluationRecord, 'id'> & { id?: string },
): Promise<EvaluationRecord> {
  return send<EvaluationRecord>(
    'POST',
    `${coursePath(courseId)}/evaluations`,
    evaluation,
    'The backend could not save the evaluation.',
  );
}

export function deleteEvaluation(
  courseId: string,
  evaluationId: string,
): Promise<{ courseId: string; deleted: number }> {
  return send<{ courseId: string; deleted: number }>(
    'DELETE',
    `${coursePath(courseId)}/evaluations/${encodeURIComponent(evaluationId)}`,
    undefined,
    'The backend could not delete the evaluation.',
  );
}

export function deleteAllEvaluations(
  courseId: string,
): Promise<{ courseId: string; deleted: number }> {
  return send<{ courseId: string; deleted: number }>(
    'DELETE',
    `${coursePath(courseId)}/evaluations`,
    undefined,
    'The backend could not clear evaluations for this course.',
  );
}

/* ------------------------------------------------------------------------ *
 * Model registry
 * ------------------------------------------------------------------------ */

export interface DbModelRegistryResponse extends CourseModelRegistry {
  courseId: string;
}

export function getCourseModel(
  courseId: string,
): Promise<DbModelRegistryResponse> {
  return get<DbModelRegistryResponse>(
    `${coursePath(courseId)}/model`,
    'The backend could not load the model for this course.',
  );
}

/* ------------------------------------------------------------------------ *
 * Model requests
 * ------------------------------------------------------------------------ */

export function getModelRequest(courseId: string): Promise<CourseModelRequest> {
  return get<CourseModelRequest>(
    `${coursePath(courseId)}/model-request`,
    'The backend could not load the model request for this course.',
  );
}

export function createModelRequest(
  courseId: string,
  approvedExampleCount: number,
): Promise<CourseModelRequest> {
  return send<CourseModelRequest>(
    'POST',
    `${coursePath(courseId)}/model-request`,
    { approvedExampleCount },
    'The backend could not submit the model request.',
  );
}

export function updateModelRequest(
  courseId: string,
  patch: Partial<CourseModelRequest>,
): Promise<CourseModelRequest> {
  return send<CourseModelRequest>(
    'PATCH',
    `${coursePath(courseId)}/model-request`,
    patch,
    'The backend could not update the model request.',
  );
}

/* ------------------------------------------------------------------------ *
 * Training runs
 * ------------------------------------------------------------------------ */

export interface DbTrainingRunListResponse {
  courseId: string;
  count: number;
  runs: TrainingRun[];
}

export function listTrainingRuns(
  courseId: string,
): Promise<DbTrainingRunListResponse> {
  return get<DbTrainingRunListResponse>(
    `${coursePath(courseId)}/training-runs`,
    'The backend could not load training runs for this course.',
  );
}

export function getTrainingRun(
  courseId: string,
  runId: string,
): Promise<TrainingRun> {
  return get<TrainingRun>(
    `${coursePath(courseId)}/training-runs/${encodeURIComponent(runId)}`,
    'The backend could not load this training run.',
  );
}

export interface EnqueueTrainingRunBody {
  mode: string;
  datasetRef: string;
  approvedExampleCount?: number;
  trainExamples?: number;
  validationExamples?: number;
}

export function createTrainingRun(
  courseId: string,
  body: EnqueueTrainingRunBody,
): Promise<TrainingRun> {
  return send<TrainingRun>(
    'POST',
    `${coursePath(courseId)}/training-runs`,
    body,
    'The backend could not queue a training run.',
  );
}

export function updateTrainingRun(
  courseId: string,
  runId: string,
  patch: Record<string, unknown>,
): Promise<TrainingRun> {
  return send<TrainingRun>(
    'PATCH',
    `${coursePath(courseId)}/training-runs/${encodeURIComponent(runId)}`,
    patch,
    'The backend could not update this training run.',
  );
}

export interface DbServingSessionResponse {
  /** null whenever nothing is serving, which is the usual answer. */
  session: unknown;
}

/**
 * The fine-tuned serving session the backend currently knows about.
 *
 * Not course-scoped, because a session is not: one Slurm allocation serves
 * every course whose adapter it can load. The response carries no compute
 * hostname or port — see `db_serving_sessions.public_serving_session` for why.
 */
export function getServingSession(): Promise<DbServingSessionResponse> {
  return get<DbServingSessionResponse>(
    `${DB_ROOT}/serving-session`,
    'The backend could not report whether a fine-tuned service is running.',
  );
}
