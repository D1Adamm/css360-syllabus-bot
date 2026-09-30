import { beforeEach, describe, expect, it, vi } from 'vitest';

const {
  getCourseMock,
  createCourseMetadataMock,
  generateCourseIdMock,
} = vi.hoisted(() => ({
  getCourseMock: vi.fn(),
  createCourseMetadataMock: vi.fn(),
  generateCourseIdMock: vi.fn(),
}));

vi.mock('./coursesDb', () => ({
  createCourseMetadata: createCourseMetadataMock,
}));

// Any read of the course being created would go through here. Professors are
// refused (403) on a course they do not staff yet, so there must be none.
vi.mock('./dbApi', () => ({
  getCourse: getCourseMock,
}));

vi.mock('./courseId', async () => {
  const actual = await vi.importActual<typeof import('./courseId')>('./courseId');
  return {
    ...actual,
    generateCourseId: generateCourseIdMock,
  };
});

import { ApiError } from './httpClient';
import { createCourse } from './createCourse';

describe('createCourse', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    createCourseMetadataMock.mockResolvedValue(undefined);
  });

  it('saves metadata as the course’s row', async () => {
    generateCourseIdMock.mockReturnValue('css-430-summer-2026-a82f');

    const result = await createCourse({
      name: 'CSS 430',
      title: 'Operating Systems',
      term: 'Summer 2026',
      instructorName: 'Ada',
    });

    expect(result.courseId).toBe('css-430-summer-2026-a82f');
    expect(createCourseMetadataMock).toHaveBeenCalledWith(
      'css-430-summer-2026-a82f',
      expect.objectContaining({
        name: 'CSS 430',
        title: 'Operating Systems',
        term: 'Summer 2026',
        instructorName: 'Ada',
        syllabusStatus: 'not_uploaded',
        syllabusFileName: '',
        syllabusType: '',
        chunkCount: 0,
      }),
    );
    // Course scoping is the id itself now: it is what every request path and
    // every table key is built from, so the created course must carry the id
    // that was generated for it.
    expect(result.courseId).toBe('css-430-summer-2026-a82f');
    expect(createCourseMetadataMock.mock.calls[0][0]).toBe(result.courseId);
  });

  it('never reads the course before creating it', async () => {
    generateCourseIdMock.mockReturnValue('css-430-summer-2026-a82f');

    await createCourse({ name: 'CSS 430', title: 'Operating Systems', term: 'Summer 2026' });

    expect(getCourseMock).not.toHaveBeenCalled();
    expect(createCourseMetadataMock).toHaveBeenCalledTimes(1);
  });

  it('regenerates the course id when the backend reports it taken (409)', async () => {
    generateCourseIdMock
      .mockReturnValueOnce('css-430-summer-2026-aaaa')
      .mockReturnValueOnce('css-430-summer-2026-bbbb');
    createCourseMetadataMock
      .mockRejectedValueOnce(new ApiError('Course "css-430-summer-2026-aaaa" already exists.', 409))
      .mockResolvedValueOnce(undefined);

    const result = await createCourse({
      name: 'CSS 430',
      title: 'Operating Systems',
      term: 'Summer 2026',
    });

    expect(result.courseId).toBe('css-430-summer-2026-bbbb');
    expect(createCourseMetadataMock).toHaveBeenCalledTimes(2);
    expect(createCourseMetadataMock).toHaveBeenLastCalledWith(
      'css-430-summer-2026-bbbb',
      expect.any(Object),
    );
  });

  it('surfaces any other failure at once instead of retrying', async () => {
    generateCourseIdMock.mockReturnValue('css-430-summer-2026-a82f');
    createCourseMetadataMock.mockRejectedValue(
      new ApiError('You do not have access to this.', 403),
    );

    await expect(
      createCourse({ name: 'CSS 430', title: 'Operating Systems', term: 'Summer 2026' }),
    ).rejects.toMatchObject({ status: 403 });
    expect(createCourseMetadataMock).toHaveBeenCalledTimes(1);
  });

  it('throws when unique ids cannot be allocated', async () => {
    generateCourseIdMock.mockReturnValue('css-430-summer-2026-zzzz');
    createCourseMetadataMock.mockRejectedValue(new ApiError('already exists', 409));

    await expect(
      createCourse({
        name: 'CSS 430',
        title: 'Operating Systems',
        term: 'Summer 2026',
      }),
    ).rejects.toThrow(/unique course id/);

    expect(createCourseMetadataMock).toHaveBeenCalledTimes(8);
  });
});
