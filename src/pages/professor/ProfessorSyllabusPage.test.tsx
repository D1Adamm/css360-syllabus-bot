/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { CourseProvider } from '../../context/CourseContext';
import type { CourseMetadata } from '../../types';

const { fetchCourseSyllabusTextMock, uploadCourseSyllabusMock, subscribeToCourseMetadataMock } =
  vi.hoisted(() => ({
    fetchCourseSyllabusTextMock: vi.fn(),
    uploadCourseSyllabusMock: vi.fn(),
    subscribeToCourseMetadataMock: vi.fn(),
  }));

vi.mock('../../lib/api', async () => {
  const actual = await vi.importActual<typeof import('../../lib/api')>('../../lib/api');
  return {
    ...actual,
    fetchCourseSyllabusText: fetchCourseSyllabusTextMock,
    uploadCourseSyllabus: uploadCourseSyllabusMock,
  };
});

vi.mock('../../lib/coursesDb', async () => {
  const actual = await vi.importActual<typeof import('../../lib/coursesDb')>(
    '../../lib/coursesDb',
  );
  return { ...actual, subscribeToCourseMetadata: subscribeToCourseMetadataMock };
});

import { ProfessorSyllabusPage } from './ProfessorSyllabusPage';

const COURSE = 'css-430-fall-2026-k7q2';

function metadata(syllabusStatus: CourseMetadata['syllabusStatus']): CourseMetadata {
  return {
    name: 'CSS 430',
    title: 'Operating Systems',
    term: 'Fall 2026',
    instructorName: '',
    createdAt: '2026-09-30T00:00:00.000Z',
    syllabusStatus,
    syllabusFileName: null,
    syllabusType: null,
    chunkCount: 0,
  };
}

function renderPage() {
  return render(
    <CourseProvider courseId={COURSE}>
      <ProfessorSyllabusPage />
    </CourseProvider>,
  );
}

describe('ProfessorSyllabusPage', () => {
  beforeEach(() => {
    fetchCourseSyllabusTextMock.mockReset();
    uploadCourseSyllabusMock.mockReset();
    subscribeToCourseMetadataMock.mockReset();
  });

  afterEach(() => {
    cleanup();
  });

  it('recovers a course whose first upload failed, then shows the new syllabus', async () => {
    let status: CourseMetadata['syllabusStatus'] = 'index_failed';
    subscribeToCourseMetadataMock.mockImplementation(
      (_courseId: string, onData: (value: CourseMetadata) => void) => {
        onData(metadata(status));
        return () => undefined;
      },
    );
    const { ApiError } = await import('../../lib/httpClient');
    fetchCourseSyllabusTextMock
      .mockRejectedValueOnce(new ApiError('not found', 404))
      .mockResolvedValue({ courseId: COURSE, text: 'CSS 430 Fall syllabus', characterCount: 21 });
    uploadCourseSyllabusMock.mockImplementation(async () => {
      status = 'indexed';
      return {
        courseId: COURSE,
        syllabusFileName: 'fall.pdf',
        syllabusType: 'pdf',
        syllabusStatus: 'indexed',
        fileSize: 16,
        characterCount: 21,
        chunkCount: 3,
        replaced: false,
      };
    });

    renderPage();

    expect(
      await screen.findByText('The last syllabus upload could not be processed'),
    ).toBeInTheDocument();
    expect(await screen.findByText('No syllabus yet')).toBeInTheDocument();

    const file = new File(['%PDF-1.4'], 'fall.pdf', { type: 'application/pdf' });
    fireEvent.change(screen.getByLabelText(/Syllabus file/), { target: { files: [file] } });
    fireEvent.click(screen.getByRole('button', { name: 'Upload syllabus' }));

    expect(await screen.findByText('CSS 430 Fall syllabus')).toBeInTheDocument();
    expect(uploadCourseSyllabusMock).toHaveBeenCalledWith(COURSE, file);
    // The status is read again, so the panel now offers a replacement.
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Replace syllabus' })).toBeInTheDocument(),
    );
  });
});
