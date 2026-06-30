export interface BaseModelGenerateResponse {
  answer: string;
  model: string;
  responseType: 'base';
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

export async function generateBaseModel(
  question: string,
): Promise<BaseModelGenerateResponse> {
  const baseUrl = getApiBaseUrl();

  if (!baseUrl) {
    throw new ApiError(
      'API base URL is not configured. Set VITE_API_BASE_URL in your .env file.',
    );
  }

  let response: Response;

  try {
    response = await fetch(`${baseUrl}/base-model/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question }),
    });
  } catch {
    throw new ApiError(
      'Could not reach the backend. Make sure the FastAPI server is running.',
    );
  }

  if (!response.ok) {
    let detail = 'The backend could not generate a base model response.';

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

  return (await response.json()) as BaseModelGenerateResponse;
}
