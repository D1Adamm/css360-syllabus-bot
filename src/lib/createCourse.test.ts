import { beforeEach, describe, expect, it, vi } from 'vitest';
import { getCourseMetadataPath } from './coursePaths';

const {
  courseExistsMock,
  createCourseMetadataMock,
  generateCourseIdMock,
} = vi.hoisted(() => ({
  courseExistsMock: vi.fn(),
  createCourseMetadataMock: vi.fn(),
  generateCourseIdMock: vi.fn(),
}));

vi.mock('./coursesDb', () => ({
  courseExists: courseExistsMock,
  createCourseMetadata: createCourseMetadataMock,
}));

vi.mock('./courseId', async () => {
  const actual = await vi.importActual<typeof import('./courseId')>('./courseId');
  return {
    ...actual,
    generateCourseId: generateCourseIdMock,
  };
});

import { createCourse } from './createCourse';

describe('createCourse', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    createCourseMetadataMock.mockResolvedValue(undefined);
  });

  it('saves metadata to courses/{courseId}/metadata', async () => {
    generateCourseIdMock.mockReturnValue('css-430-summer-2026-a82f');
    courseExistsMock.mockResolvedValue(false);

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
    expect(getCourseMetadataPath(result.courseId)).toBe(
      'courses/css-430-summer-2026-a82f/metadata',
    );
  });

  it('regenerates the course id when a collision is detected', async () => {
    generateCourseIdMock
      .mockReturnValueOnce('css-430-summer-2026-aaaa')
      .mockReturnValueOnce('css-430-summer-2026-bbbb');
    courseExistsMock.mockResolvedValueOnce(true).mockResolvedValueOnce(false);

    const result = await createCourse({
      name: 'CSS 430',
      title: 'Operating Systems',
      term: 'Summer 2026',
    });

    expect(result.courseId).toBe('css-430-summer-2026-bbbb');
    expect(courseExistsMock).toHaveBeenCalledTimes(2);
    expect(createCourseMetadataMock).toHaveBeenCalledTimes(1);
    expect(createCourseMetadataMock).toHaveBeenCalledWith(
      'css-430-summer-2026-bbbb',
      expect.any(Object),
    );
  });

  it('throws when unique ids cannot be allocated', async () => {
    generateCourseIdMock.mockReturnValue('css-430-summer-2026-zzzz');
    courseExistsMock.mockResolvedValue(true);

    await expect(
      createCourse({
        name: 'CSS 430',
        title: 'Operating Systems',
        term: 'Summer 2026',
      }),
    ).rejects.toThrow(/unique course id/);

    expect(createCourseMetadataMock).not.toHaveBeenCalled();
  });
});
