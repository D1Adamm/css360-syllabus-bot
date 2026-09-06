/**
 * The one place the browser talks HTTP to the backend.
 *
 * `api.ts`, `dbApi.ts` and `adminApi.ts` each used to carry their own copy of
 * the same fetch wrapper. That was tolerable while a request needed nothing but
 * a URL. Authentication changes that: every request now has to send the session
 * cookies, every state-changing request has to carry the CSRF header, and a
 * 401 has to be noticed in one place so the session state can flip to
 * signed-out instead of every page discovering it separately. Three copies of
 * that logic would drift; one cannot.
 *
 * Paths are relative to `VITE_API_BASE_URL`, which carries the `/api` prefix
 * (`/api` locally through the Vite proxy, `https://aiswe.uwb.edu/api` on the
 * VM). Callers write only the part after it.
 */

export class ApiError extends Error {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

/**
 * Cross-site request forgery protection, browser half.
 *
 * Sessions live in `SameSite=Lax` cookies, which already stops a cross-site
 * form from carrying them on a POST. This header is the second, independent
 * check: a cross-site request cannot set a custom header without a CORS
 * preflight, and the backend refuses cookie-authenticated mutations that lack
 * it. No per-form token has to be fetched, stored, or kept in sync.
 */
export const CSRF_HEADER_NAME = 'X-Requested-With';
export const CSRF_HEADER_VALUE = 'SyllabusModelLab';

export const UNREACHABLE_MESSAGE = 'The service could not be reached.';
export const NOT_CONFIGURED_MESSAGE = 'The service is not configured.';

type HttpMethod = 'GET' | 'POST' | 'PATCH' | 'PUT' | 'DELETE';

const SAFE_METHODS: ReadonlySet<HttpMethod> = new Set<HttpMethod>(['GET']);

export function getApiBaseUrl(): string | null {
  const value = import.meta.env.VITE_API_BASE_URL;

  if (typeof value !== 'string' || value.trim() === '') {
    return null;
  }

  return value.trim().replace(/\/$/, '');
}

type UnauthorizedListener = () => void;

const unauthorizedListeners = new Set<UnauthorizedListener>();

/**
 * Be told whenever the backend answers 401.
 *
 * The session provider uses this to drop its cached identity the moment a
 * request is refused, so an expired or revoked session turns into the sign-in
 * screen on the next click rather than into a page of error banners.
 */
export function onUnauthorized(listener: UnauthorizedListener): () => void {
  unauthorizedListeners.add(listener);
  return () => {
    unauthorizedListeners.delete(listener);
  };
}

function notifyUnauthorized(): void {
  for (const listener of unauthorizedListeners) {
    try {
      listener();
    } catch {
      // A listener that throws must not stop the others from hearing.
    }
  }
}

export interface RequestOptions {
  method?: HttpMethod;
  /** Sent as JSON with the matching content type. */
  json?: unknown;
  /** Sent as-is; for multipart uploads, where the browser sets the type. */
  body?: BodyInit;
  /** Shown when the backend refuses without a usable `detail`. */
  fallbackErrorMessage: string;
  /** Shown when no response arrives at all. */
  unreachableMessage?: string;
  /**
   * Give up after this long and throw an `ApiError` saying so.
   *
   * Off by default: the comparison page legitimately waits on a CPU-bound
   * model for longer than any fixed limit would allow. A caller that has
   * arranged for the backend to answer quickly — the admin diagnostics, which
   * ask for status rather than waiting on work — sets this so a proxy that
   * holds the socket open cannot leave a control spinning forever.
   */
  timeoutMs?: number;
}

export function timeoutMessage(timeoutMs: number): string {
  return `No response after ${Math.round(timeoutMs / 1000)} seconds. The request was stopped.`;
}

async function readDetail(response: Response, fallback: string): Promise<string> {
  try {
    const errorBody = (await response.json()) as { detail?: unknown };
    if (typeof errorBody.detail === 'string' && errorBody.detail.trim() !== '') {
      return errorBody.detail;
    }
  } catch {
    // Keep the fallback when the error body is not JSON.
  }
  return fallback;
}

/**
 * One request, one place to get the cross-cutting parts right.
 *
 *   - cookies travel with every request (`credentials: 'include'`), which is
 *     what carries the staff and participant sessions;
 *   - state-changing requests carry the CSRF header;
 *   - a network failure and a refused request are distinguishable by message,
 *     exactly as the three wrappers this replaces already promised;
 *   - a 401 is announced to whoever is listening before it is thrown.
 */
export async function requestJson<T>(path: string, options: RequestOptions): Promise<T> {
  const baseUrl = getApiBaseUrl();

  if (!baseUrl) {
    throw new ApiError(NOT_CONFIGURED_MESSAGE);
  }

  const method: HttpMethod = options.method ?? 'GET';
  const headers: Record<string, string> = {};

  if (!SAFE_METHODS.has(method)) {
    headers[CSRF_HEADER_NAME] = CSRF_HEADER_VALUE;
  }

  let body: BodyInit | undefined = options.body;
  if (options.json !== undefined) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(options.json);
  }

  const controller = options.timeoutMs ? new AbortController() : null;
  const timer =
    controller && options.timeoutMs
      ? setTimeout(() => controller.abort(), options.timeoutMs)
      : null;

  const init: RequestInit = {
    method,
    credentials: 'include',
    ...(Object.keys(headers).length > 0 ? { headers } : {}),
    ...(body === undefined ? {} : { body }),
    ...(controller ? { signal: controller.signal } : {}),
  };

  let response: Response;

  try {
    response = await fetch(`${baseUrl}${path}`, init);
  } catch {
    if (controller?.signal.aborted && options.timeoutMs) {
      throw new ApiError(timeoutMessage(options.timeoutMs));
    }
    throw new ApiError(options.unreachableMessage ?? UNREACHABLE_MESSAGE);
  } finally {
    if (timer !== null) {
      clearTimeout(timer);
    }
  }

  if (!response.ok) {
    const detail = await readDetail(response, options.fallbackErrorMessage);
    if (response.status === 401) {
      notifyUnauthorized();
    }
    throw new ApiError(detail, response.status);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export function getJson<T>(
  path: string,
  fallbackErrorMessage: string,
  unreachableMessage?: string,
): Promise<T> {
  return requestJson<T>(path, { method: 'GET', fallbackErrorMessage, unreachableMessage });
}

export function postJson<T>(
  path: string,
  json: unknown,
  fallbackErrorMessage: string,
  unreachableMessage?: string,
): Promise<T> {
  return requestJson<T>(path, {
    method: 'POST',
    json,
    fallbackErrorMessage,
    unreachableMessage,
  });
}

export function sendJson<T>(
  method: 'POST' | 'PATCH' | 'PUT' | 'DELETE',
  path: string,
  json: unknown,
  fallbackErrorMessage: string,
  unreachableMessage?: string,
): Promise<T> {
  return requestJson<T>(path, {
    method,
    ...(json === undefined ? {} : { json }),
    fallbackErrorMessage,
    unreachableMessage,
  });
}
