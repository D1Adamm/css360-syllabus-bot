/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { CourseProvider } from '../../context/CourseContext';
import { ApiError } from '../../lib/httpClient';
import type { CourseMetadata } from '../../types';

const {
  fetchCourseSyllabusTextMock,
  uploadCourseSyllabusMock,
  hasCourseSyllabusFileMock,
  subscribeToCourseMetadataMock,
} = vi.hoisted(() => ({
  fetchCourseSyllabusTextMock: vi.fn(),
  uploadCourseSyllabusMock: vi.fn(),
  hasCourseSyllabusFileMock: vi.fn(),
  subscribeToCourseMetadataMock: vi.fn(),
}));

vi.mock('../../lib/api', async () => {
  const actual = await vi.importActual<typeof import('../../lib/api')>('../../lib/api');
  return {
    ...actual,
    fetchCourseSyllabusText: fetchCourseSyllabusTextMock,
    uploadCourseSyllabus: uploadCourseSyllabusMock,
    hasCourseSyllabusFile: hasCourseSyllabusFileMock,
    courseSyllabusFileUrl: (courseId: string, download = false) =>
      `/api/courses/${courseId}/syllabus/file${download ? '?download=1' : ''}`,
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
const LONG_TEXT = `Course Overview\n\n${'Operating systems, processes and threads. '.repeat(400)}`;

let metadataStatus: CourseMetadata['syllabusStatus'];

function metadata(): CourseMetadata {
  return {
    name: 'CSS 430',
    title: 'Operating Systems',
    term: 'Fall 2026',
    instructorName: '',
    createdAt: '2026-09-30T00:00:00.000Z',
    syllabusStatus: metadataStatus,
    syllabusFileName: metadataStatus === 'indexed' ? 'CSS430-Fall2026.pdf' : null,
    syllabusType: metadataStatus === 'indexed' ? 'pdf' : null,
    chunkCount: metadataStatus === 'indexed' ? 42 : 0,
  };
}

function renderPage() {
  return render(
    <CourseProvider courseId={COURSE}>
      <ProfessorSyllabusPage />
    </CourseProvider>,
  );
}

function chooseFile(name = 'fall-v2.pdf') {
  const file = new File(['%PDF-1.4'], name, { type: 'application/pdf' });
  fireEvent.change(screen.getByLabelText(/Syllabus file/), { target: { files: [file] } });
  return file;
}

describe('ProfessorSyllabusPage', () => {
  beforeEach(() => {
    metadataStatus = 'indexed';
    fetchCourseSyllabusTextMock.mockReset();
    uploadCourseSyllabusMock.mockReset();
    hasCourseSyllabusFileMock.mockReset();
    hasCourseSyllabusFileMock.mockResolvedValue(true);
    subscribeToCourseMetadataMock.mockReset();
    subscribeToCourseMetadataMock.mockImplementation(
      (_courseId: string, onData: (value: CourseMetadata) => void) => {
        onData(metadata());
        return () => undefined;
      },
    );
    fetchCourseSyllabusTextMock.mockResolvedValue({
      courseId: COURSE,
      text: LONG_TEXT,
      characterCount: LONG_TEXT.length,
    });
  });

  afterEach(() => {
    cleanup();
  });

  it('shows a compact card for the current syllabus, with the original file', async () => {
    renderPage();

    expect(await screen.findByText('CSS430-Fall2026.pdf')).toBeInTheDocument();
    expect(screen.getByText('PDF · Ready · 42 sections indexed')).toBeInTheDocument();
    const view = await screen.findByRole('link', { name: 'View syllabus' });
    expect(view).toHaveAttribute('href', `/api/courses/${COURSE}/syllabus/file`);
    expect(view).toHaveAttribute('target', '_blank');
    expect(screen.getByRole('link', { name: 'Download original' })).toHaveAttribute(
      'href',
      `/api/courses/${COURSE}/syllabus/file?download=1`,
    );

    // Compact: no extracted text and no upload area until asked for.
    expect(fetchCourseSyllabusTextMock).not.toHaveBeenCalled();
    expect(screen.queryByTestId('extracted-syllabus-text')).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/Syllabus file/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Replace syllabus' })).toBeInTheDocument();
  });

  it('keeps the extracted text collapsed until opened, then shows it in a bounded box', async () => {
    renderPage();
    const toggle = await screen.findByRole('button', { name: 'Show extracted text' });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');

    fireEvent.click(toggle);
    const text = await screen.findByTestId('extracted-syllabus-text');
    expect(text).toHaveClass('syllabus-extracted__text');
    expect(text.textContent).toBe(LONG_TEXT);
    expect(fetchCourseSyllabusTextMock).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'Hide extracted text' })).toHaveAttribute(
      'aria-expanded',
      'true',
    );

    fireEvent.click(screen.getByRole('button', { name: 'Hide extracted text' }));
    expect(screen.queryByTestId('extracted-syllabus-text')).not.toBeInTheDocument();
  });

  it('reveals the upload panel on Replace, replaces, and refreshes what it shows', async () => {
    uploadCourseSyllabusMock.mockResolvedValue({
      courseId: COURSE,
      syllabusFileName: 'fall-v2.pdf',
      syllabusType: 'pdf',
      syllabusStatus: 'indexed',
      fileSize: 8,
      characterCount: 900,
      chunkCount: 40,
      replaced: true,
    });
    renderPage();

    fireEvent.click(await screen.findByRole('button', { name: 'Replace syllabus' }));
    const file = chooseFile();
    const submit = screen
      .getAllByRole('button', { name: 'Replace syllabus' })
      .find((button) => button.getAttribute('type') === 'submit');
    fireEvent.click(submit!);

    expect(await screen.findByText(/Syllabus replaced/)).toBeInTheDocument();
    expect(uploadCourseSyllabusMock).toHaveBeenCalledWith(COURSE, file);
    await waitFor(() => expect(hasCourseSyllabusFileMock).toHaveBeenCalledTimes(2));

    fireEvent.click(screen.getByRole('button', { name: 'Done' }));
    expect(screen.queryByLabelText(/Syllabus file/)).not.toBeInTheDocument();
  });

  it('a failed replacement says the current syllabus is still in use', async () => {
    uploadCourseSyllabusMock.mockRejectedValue(
      new ApiError('Ollama is unavailable for embeddings.', 503),
    );
    renderPage();

    fireEvent.click(await screen.findByRole('button', { name: 'Replace syllabus' }));
    chooseFile();
    const submit = screen
      .getAllByRole('button', { name: 'Replace syllabus' })
      .find((button) => button.getAttribute('type') === 'submit');
    fireEvent.click(submit!);

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Your current syllabus is still in use.',
    );
    expect(screen.getByText('CSS430-Fall2026.pdf')).toBeInTheDocument();
  });

  it('opens the upload panel straight away when the first upload failed', async () => {
    metadataStatus = 'index_failed';
    hasCourseSyllabusFileMock.mockResolvedValue(false);
    renderPage();

    expect(
      await screen.findByText('The last syllabus upload could not be processed'),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/Syllabus file/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Upload syllabus' })).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'View syllabus' })).not.toBeInTheDocument();
  });

  it('explains a course whose original file was never stored', async () => {
    hasCourseSyllabusFileMock.mockResolvedValue(false);
    renderPage();

    expect(
      await screen.findByText(/The original file is not stored for this course/),
    ).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'View syllabus' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Show extracted text' })).toBeInTheDocument();
  });
});
