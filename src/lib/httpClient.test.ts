import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

/**
 * The cross-cutting guarantees every API module now inherits.
 *
 * Each of these was previously a property of three separate fetch wrappers,
 * or of none of them. They are asserted here once because the session design
 * depends on them: cookies must travel, mutations must carry the CSRF header,
 * and a 401 must reach the session provider.
 */

const fetchMock = vi.fn();

beforeEach(() => {
  vi.resetModules();
  vi.stubGlobal('fetch', fetchMock);
  vi.stubEnv('VITE_API_BASE_URL', 'https://aiswe.uwb.edu/api');
  fetchMock.mockReset();
  fetchMock.mockResolvedValue({ ok: true, status: 200, json: async () => ({ ok: true }) });
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

function requestInit(callIndex = 0): RequestInit {
  return fetchMock.mock.calls[callIndex][1] as RequestInit;
}

function headersOf(init: RequestInit): Record<string, string> {
  return (init.headers ?? {}) as Record<string, string>;
}

describe('httpClient', () => {
  it('sends credentials on every request so the session cookies travel', async () => {
    const { getJson, postJson } = await import('./httpClient');
    await getJson('/auth/session', 'failed');
    await postJson('/db/courses', { name: 'x' }, 'failed');

    expect(requestInit(0).credentials).toBe('include');
    expect(requestInit(1).credentials).toBe('include');
  });

  it('adds the CSRF header to mutations and not to reads', async () => {
    const { CSRF_HEADER_NAME, CSRF_HEADER_VALUE, getJson, sendJson } = await import(
      './httpClient'
    );
    await getJson('/db/courses', 'failed');
    await sendJson('POST', '/db/courses', { name: 'x' }, 'failed');
    await sendJson('PATCH', '/db/courses/c', { name: 'y' }, 'failed');
    await sendJson('DELETE', '/db/courses/c/seeds/s', undefined, 'failed');

    expect(headersOf(requestInit(0))[CSRF_HEADER_NAME]).toBeUndefined();
    expect(headersOf(requestInit(1))[CSRF_HEADER_NAME]).toBe(CSRF_HEADER_VALUE);
    expect(headersOf(requestInit(2))[CSRF_HEADER_NAME]).toBe(CSRF_HEADER_VALUE);
    expect(headersOf(requestInit(3))[CSRF_HEADER_NAME]).toBe(CSRF_HEADER_VALUE);
  });

  it('serialises JSON bodies with the JSON content type and leaves multipart alone', async () => {
    const { requestJson } = await import('./httpClient');
    const form = new FormData();
    await requestJson('/db/courses', { method: 'POST', json: { a: 1 }, fallbackErrorMessage: 'f' });
    await requestJson('/courses/c/syllabus', {
      method: 'POST',
      body: form,
      fallbackErrorMessage: 'f',
    });

    expect(headersOf(requestInit(0))['Content-Type']).toBe('application/json');
    expect(requestInit(0).body).toBe('{"a":1}');
    expect(headersOf(requestInit(1))['Content-Type']).toBeUndefined();
    expect(requestInit(1).body).toBe(form);
  });

  it('announces a 401 to listeners before throwing it', async () => {
    const { ApiError, getJson, onUnauthorized } = await import('./httpClient');
    fetchMock.mockResolvedValue({
      ok: false,
      status: 401,
      json: async () => ({ detail: 'Sign in to continue.' }),
    });
    const listener = vi.fn();
    const stop = onUnauthorized(listener);

    await expect(getJson('/db/courses', 'failed')).rejects.toMatchObject({
      status: 401,
      message: 'Sign in to continue.',
    });
    expect(listener).toHaveBeenCalledTimes(1);

    stop();
    await expect(getJson('/db/courses', 'failed')).rejects.toBeInstanceOf(ApiError);
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it('does not announce a 403: the caller is signed in and simply not allowed', async () => {
    const { getJson, onUnauthorized } = await import('./httpClient');
    fetchMock.mockResolvedValue({ ok: false, status: 403, json: async () => ({}) });
    const listener = vi.fn();
    onUnauthorized(listener);

    await expect(getJson('/admin/users', 'Not allowed.')).rejects.toMatchObject({
      status: 403,
      message: 'Not allowed.',
    });
    expect(listener).not.toHaveBeenCalled();
  });

  it('reports an unreachable service with the caller-supplied message', async () => {
    const { getJson } = await import('./httpClient');
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'));

    await expect(getJson('/health', 'failed', 'Backend down.')).rejects.toMatchObject({
      message: 'Backend down.',
    });
  });
});
