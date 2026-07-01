export interface BaseModelGenerateResponse {
  answer: string;
  model: string;
  responseType: 'base';
}

export interface RagGenerateSource {
  section: string;
}

export interface RagRetrieveResult {
  chunkId: string;
  section: string;
  text: string;
  score: number;
}

export interface RagGenerateResponse {
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

async function postJson<T>(
  path: string,
  body: unknown,
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
    response = await fetch(`${baseUrl}${path}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
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

export async function generateBaseModel(
  question: string,
): Promise<BaseModelGenerateResponse> {
  return postJson<BaseModelGenerateResponse>(
    '/base-model/generate',
    { question },
    'The backend could not generate a base model response.',
    'Could not reach the backend. Make sure the FastAPI server is running.',
  );
}

export async function generateRag(
  question: string,
  topK = 4,
): Promise<RagGenerateResponse> {
  return postJson<RagGenerateResponse>(
    '/rag/generate',
    { question, topK },
    'The backend could not generate a RAG response.',
    'Could not reach the backend. Make sure the FastAPI server is running and Ollama is available.',
  );
}
