import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, fetchCourseSyllabusText } from './api';

describe('fetchCourseSyllabusText', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('requests syllabus text for the provided courseId', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://127.0.0.1:8001');

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        courseId: 'css-430-summer-2026-ibce',
        text: 'CSS 430 syllabus body',
        characterCount: 21,
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    const result = await fetchCourseSyllabusText('css-430-summer-2026-ibce');

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(
      'http://127.0.0.1:8001/api/courses/css-430-summer-2026-ibce/syllabus/text',
    );
    expect(options.method).toBe('GET');
    expect(result.courseId).toBe('css-430-summer-2026-ibce');
    expect(result.text).toBe('CSS 430 syllabus body');
  });

  it('surfaces backend detail for missing syllabus text', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://127.0.0.1:8001');
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        json: async () => ({
          detail: 'Extracted syllabus text was not found for this course.',
        }),
      }),
    );

    await expect(fetchCourseSyllabusText('missing-course')).rejects.toMatchObject({
      name: 'ApiError',
      status: 404,
      message: 'Extracted syllabus text was not found for this course.',
    } satisfies Partial<ApiError>);
  });
});
