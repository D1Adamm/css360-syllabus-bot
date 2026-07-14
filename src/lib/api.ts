export interface BaseModelGenerateResponse {
  answer: string;
  model: string;
  responseType: 'base';
  courseId?: string;
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

export async function generateRag(
  courseId: string,
  question: string,
  topK = 3,
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
