export interface BaseModelGenerateResponse {
  answer: string;
  model: string;
  responseType: 'base';
  courseId?: string;
}

export interface FineTunedGenerateResponse {
  answer: string;
  model: string;
  responseType: 'fineTuned';
  courseId?: string;
  adapterLoaded: boolean;
  generationSeconds?: number | null;
}

export interface FineTunedRagGenerateResponse {
  courseId: string;
  answer: string;
  model: string;
  sources: RagGenerateSource[];
  retrievedChunks: RagRetrieveResult[];
  responseType: 'fineTunedRag';
  adapterLoaded: boolean;
  generationSeconds?: number | null;
}

export interface RagGenerateSource {
  chunkId: string;
  sectionTitle: string;
  text: string;
  score: number;
}

export interface RagRetrieveResult {
  chunkId: string;
  section: string;
  text: string;
  score: number;
}

export interface RagGenerateResponse {
  courseId: string;
  answer: string;
  model: string;
  sources: RagGenerateSource[];
  retrievedChunks: RagRetrieveResult[];
  responseType: 'rag';
}

export class ApiError extends Error {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

export function getApiBaseUrl(): string | null {
  const value = import.meta.env.VITE_API_BASE_URL;

  if (typeof value !== 'string' || value.trim() === '') {
    return null;
  }

  return value.trim().replace(/\/$/, '');
}

async function requestJson<T>(
  path: string,
  init: RequestInit,
  fallbackErrorMessage: string,
  unreachableMessage: string,
): Promise<T> {
  const baseUrl = getApiBaseUrl();

  if (!baseUrl) {
    throw new ApiError(
      'API base URL is not configured. Set VITE_API_BASE_URL in your .env file.',
    );
  }

  let response: Response;

  try {
    response = await fetch(`${baseUrl}${path}`, init);
  } catch {
    throw new ApiError(unreachableMessage);
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

  return (await response.json()) as T;
}

async function postJson<T>(
  path: string,
  body: unknown,
  fallbackErrorMessage: string,
  unreachableMessage: string,
): Promise<T> {
  return requestJson<T>(
    path,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    },
    fallbackErrorMessage,
    unreachableMessage,
  );
}

async function getJson<T>(
  path: string,
  fallbackErrorMessage: string,
  unreachableMessage: string,
): Promise<T> {
  return requestJson<T>(
    path,
    { method: 'GET' },
    fallbackErrorMessage,
    unreachableMessage,
  );
}

export async function generateBaseModel(
  courseId: string,
  question: string,
): Promise<BaseModelGenerateResponse> {
  return postJson<BaseModelGenerateResponse>(
    '/base-model/generate',
    { courseId, question },
    'The backend could not generate a base model response.',
    'Could not reach the backend. Make sure the FastAPI server is running.',
  );
}

export async function generateFineTuned(
  courseId: string,
  question: string,
): Promise<FineTunedGenerateResponse> {
  return postJson<FineTunedGenerateResponse>(
    '/fine-tuned/generate',
    { courseId, question },
    'The backend could not generate a fine-tuned model response.',
    'Could not reach the backend. Make sure the FastAPI server is running and FINETUNED_SERVICE_URL is set.',
  );
}

export async function generateFineTunedRag(
  courseId: string,
  question: string,
  topK = 5,
): Promise<FineTunedRagGenerateResponse> {
  return postJson<FineTunedRagGenerateResponse>(
    '/fine-tuned-rag/generate',
    { courseId, question, topK },
    'The backend could not generate a Fine-Tuned + RAG response.',
    'Could not reach the backend. Make sure the FastAPI server is running, the course syllabus is indexed, and FINETUNED_SERVICE_URL is set.',
  );
}

export async function generateRag(
  courseId: string,
  question: string,
  topK = 5,
): Promise<RagGenerateResponse> {
  return postJson<RagGenerateResponse>(
    '/rag/generate',
    { courseId, question, topK },
    'The backend could not generate a RAG response.',
    'Could not reach the backend. Make sure the FastAPI server is running and Ollama is available.',
  );
}

export interface SyllabusUploadResponse {
  courseId: string;
  syllabusFileName: string;
  syllabusType: string;
  syllabusStatus: string;
  fileSize: number;
  characterCount: number;
  chunkCount: number;
}

export interface SyllabusTextResponse {
  courseId: string;
  text: string;
  characterCount: number;
}

export async function fetchCourseSyllabusText(
  courseId: string,
): Promise<SyllabusTextResponse> {
  return getJson<SyllabusTextResponse>(
    `/api/courses/${courseId}/syllabus/text`,
    'The backend could not load the syllabus text for this course.',
    'Could not reach the backend to load the syllabus. Make sure the FastAPI server is running.',
  );
}

export async function uploadCourseSyllabus(
  courseId: string,
  file: File,
): Promise<SyllabusUploadResponse> {
  const baseUrl = getApiBaseUrl();

  if (!baseUrl) {
    throw new ApiError(
      'API base URL is not configured. Set VITE_API_BASE_URL in your .env file.',
    );
  }

  const formData = new FormData();
  formData.append('syllabus_file', file, file.name);

  let response: Response;

  try {
    response = await fetch(`${baseUrl}/api/courses/${courseId}/syllabus`, {
      method: 'POST',
      body: formData,
    });
  } catch {
    throw new ApiError(
      'Could not reach the backend to upload the syllabus. Make sure the FastAPI server is running.',
    );
  }

  if (!response.ok) {
    let detail = 'The backend could not upload the syllabus file.';

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

  return (await response.json()) as SyllabusUploadResponse;
}

export type SeedReviewStatus = 'generated' | 'approved' | 'rejected' | 'edited';

export interface CourseSeedReviewRecord {
  id?: string | null;
  question?: string | null;
  answer?: string | null;
  instruction?: string | null;
  response?: string | null;
  category?: string | null;
  sourceSection?: string | null;
  factId?: string | null;
  evidenceQuote?: string | null;
  sourceChunkIds?: string[] | null;
  origin?: string | null;
  status?: string | null;
  reviewStatus?: string | null;
  reviewNotes?: string | null;
  originalQuestion?: string | null;
  originalAnswer?: string | null;
  /** True when the seed was human-edited; survives later approval. */
  wasEdited?: boolean | null;
  validation?: {
    score: number;
    reason: string;
    unsupportedClaims?: string[];
    components?: Record<string, number>;
  } | null;
}

export interface CourseSeedListResponse {
  courseId: string;
  count: number;
  firebasePath: string;
  seeds: CourseSeedReviewRecord[];
}

export interface SeedReviewResponse {
  courseId: string;
  seedId: string;
  seed: CourseSeedReviewRecord;
  firebasePath: string;
}

export interface SeedExportApprovedResponse {
  courseId: string;
  summary: {
    approvedCount?: number;
    exportedCount?: number;
    validatedCount?: number;
    validationPassed?: boolean;
    exportPath?: string;
    skippedCount?: number;
    existingCount?: number;
    files?: Record<string, string>;
    firebasePath?: string;
    [key: string]: unknown;
  };
}

export interface ApprovedExportStatusResponse {
  courseId: string;
  exists: boolean;
  exportPath: string;
  exampleCount: number;
  sourceFile: string;
}

export interface PrepareTrainingSplitResponse {
  courseId: string;
  summary: {
    trainExamples?: number;
    validationExamples?: number;
    totalExamples?: number;
    splitSeed?: number;
    manifest?: Record<string, unknown>;
    files?: Record<string, string>;
    [key: string]: unknown;
  };
}

export async function listCourseSeeds(
  courseId: string,
): Promise<CourseSeedListResponse> {
  return getJson<CourseSeedListResponse>(
    `/api/courses/${courseId}/seeds`,
    'The backend could not load seed examples for this course.',
    'Could not reach the backend to load seeds. Make sure the FastAPI server is running.',
  );
}

export async function reviewCourseSeed(
  courseId: string,
  seedId: string,
  body: {
    reviewStatus: SeedReviewStatus;
    question?: string;
    answer?: string;
    reviewNotes?: string;
  },
): Promise<SeedReviewResponse> {
  return postJson<SeedReviewResponse>(
    `/api/courses/${courseId}/seeds/${encodeURIComponent(seedId)}/review`,
    body,
    'The backend could not update the seed review status.',
    'Could not reach the backend to review this seed. Make sure the FastAPI server is running.',
  );
}

export async function exportApprovedCourseSeeds(
  courseId: string,
): Promise<SeedExportApprovedResponse> {
  return postJson<SeedExportApprovedResponse>(
    `/api/courses/${courseId}/seeds/export-approved`,
    {},
    'The backend could not export approved seeds.',
    'Could not reach the backend to export approved seeds. Make sure the FastAPI server is running.',
  );
}

export async function getApprovedExportStatus(
  courseId: string,
): Promise<ApprovedExportStatusResponse> {
  return getJson<ApprovedExportStatusResponse>(
    `/api/courses/${courseId}/seeds/approved-export-status`,
    'The backend could not check the approved export status.',
    'Could not reach the backend to check approved export status. Make sure the FastAPI server is running.',
  );
}

export async function prepareTrainingSplit(
  courseId: string,
): Promise<PrepareTrainingSplitResponse> {
  return postJson<PrepareTrainingSplitResponse>(
    `/api/courses/${courseId}/seeds/prepare-training-split`,
    {},
    'The backend could not prepare the training split.',
    'Could not reach the backend to prepare the training split. Make sure the FastAPI server is running.',
  );
}
