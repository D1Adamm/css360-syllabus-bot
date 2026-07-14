import { afterEach, describe, expect, it, vi } from 'vitest';
import { uploadCourseSyllabus } from './api';

describe('uploadCourseSyllabus', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('sends a multipart form request to /api/courses/{courseId}/syllabus', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://127.0.0.1:8001');

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        courseId: 'course-alpha',
        syllabusFileName: 'syllabus.pdf',
        syllabusType: 'pdf',
        syllabusStatus: 'uploaded',
        fileSize: 12,
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    const file = new File(['%PDF-sample'], 'syllabus.pdf', { type: 'application/pdf' });
    const result = await uploadCourseSyllabus('course-alpha', file);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://127.0.0.1:8001/api/courses/course-alpha/syllabus');
    expect(options.method).toBe('POST');
    expect(options.body).toBeInstanceOf(FormData);
    const formData = options.body as FormData;
    expect(formData.get('syllabus_file')).toBeInstanceOf(File);
    expect(result.syllabusStatus).toBe('uploaded');
  });
});
