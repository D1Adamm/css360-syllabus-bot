/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { ApiError } from '../lib/api';

const {
  createCourseMock,
  uploadCourseSyllabusMock,
  updateCourseMetadataMock,
} = vi.hoisted(() => ({
  createCourseMock: vi.fn(),
  uploadCourseSyllabusMock: vi.fn(),
  updateCourseMetadataMock: vi.fn(),
}));

vi.mock('../lib/firebase', () => ({
  app: {},
  database: { name: 'mock-db' },
}));

vi.mock('../lib/createCourse', () => ({
  createCourse: createCourseMock,
}));

vi.mock('../lib/api', async () => {
  const actual = await vi.importActual<typeof import('../lib/api')>('../lib/api');
  return {
    ...actual,
    uploadCourseSyllabus: uploadCourseSyllabusMock,
  };
});

vi.mock('../lib/coursesDb', async () => {
  const actual = await vi.importActual<typeof import('../lib/coursesDb')>(
    '../lib/coursesDb',
  );
  return {
    ...actual,
    updateCourseMetadata: updateCourseMetadataMock,
  };
});

import { CreateCoursePage } from '../pages/CreateCoursePage';

function LocationProbe() {
  const location = useLocation();
  return <div data-testid="location">{location.pathname}</div>;
}

function renderCreateCoursePage() {
  return render(
    <MemoryRouter initialEntries={['/create-course']}>
      <Routes>
        <Route path="/create-course" element={<CreateCoursePage />} />
        <Route path="/course/:courseId/home" element={<div>Course home</div>} />
      </Routes>
      <LocationProbe />
    </MemoryRouter>,
  );
}

function fillRequiredTextFields() {
  fireEvent.change(screen.getByLabelText(/Course name or code/), {
    target: { value: 'CSS 430' },
  });
  fireEvent.change(screen.getByLabelText(/Course title/), {
    target: { value: 'Operating Systems' },
  });
  fireEvent.change(screen.getByLabelText(/^Term/), {
    target: { value: 'Summer 2026' },
  });
}

function selectSyllabusFile(name = 'css430-syllabus.pdf', type = 'application/pdf') {
  const file = new File(['%PDF-1.4 sample'], name, { type });
  fireEvent.change(screen.getByLabelText(/Syllabus file/), {
    target: { files: [file] },
  });
  return file;
}

describe('CreateCoursePage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    updateCourseMetadataMock.mockResolvedValue(undefined);
  });

  afterEach(() => {
    cleanup();
  });

  it('requires name, title, term, and syllabus file before saving', async () => {
    renderCreateCoursePage();

    fireEvent.click(screen.getByRole('button', { name: 'Create course' }));

    expect(await screen.findByText('Course name or code is required.')).toBeInTheDocument();
    expect(screen.getByText('Course title is required.')).toBeInTheDocument();
    expect(screen.getByText('Term is required.')).toBeInTheDocument();
    expect(
      screen.getByText('A PDF or TXT syllabus file is required.'),
    ).toBeInTheDocument();
    expect(createCourseMock).not.toHaveBeenCalled();
    expect(uploadCourseSyllabusMock).not.toHaveBeenCalled();
  });

  it('uploads multipart syllabus and updates Firebase metadata before redirecting', async () => {
    createCourseMock.mockResolvedValue({
      courseId: 'css-430-summer-2026-a82f',
      metadata: {
        name: 'CSS 430',
        title: 'Operating Systems',
        term: 'Summer 2026',
        instructorName: '',
        createdAt: '2026-01-01T00:00:00.000Z',
        syllabusStatus: 'not_uploaded',
        syllabusFileName: '',
        syllabusType: '',
        chunkCount: 0,
      },
    });
    uploadCourseSyllabusMock.mockResolvedValue({
      courseId: 'css-430-summer-2026-a82f',
      syllabusFileName: 'css430-syllabus.pdf',
      syllabusType: 'pdf',
      syllabusStatus: 'extracted',
      fileSize: 16,
      characterCount: 18000,
    });

    const view = renderCreateCoursePage();
    fillRequiredTextFields();
    const file = selectSyllabusFile();
    fireEvent.click(screen.getByRole('button', { name: 'Create course' }));

    await waitFor(() => {
      expect(createCourseMock).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(uploadCourseSyllabusMock).toHaveBeenCalledWith(
        'css-430-summer-2026-a82f',
        file,
      );
    });
    await waitFor(() => {
      expect(updateCourseMetadataMock).toHaveBeenCalledWith(
        'css-430-summer-2026-a82f',
        expect.objectContaining({
          syllabusStatus: 'extracted',
          syllabusFileName: 'css430-syllabus.pdf',
          syllabusType: 'pdf',
          chunkCount: 0,
        }),
      );
    });
    await waitFor(() => {
      expect(view.getByTestId('location')).toHaveTextContent(
        '/course/css-430-summer-2026-a82f/home',
      );
    });
  });

  it('shows a readable upload error and does not redirect on failure', async () => {
    createCourseMock.mockResolvedValue({
      courseId: 'css-430-summer-2026-a82f',
      metadata: {
        name: 'CSS 430',
        title: 'Operating Systems',
        term: 'Summer 2026',
        instructorName: '',
        createdAt: '2026-01-01T00:00:00.000Z',
        syllabusStatus: 'not_uploaded',
        syllabusFileName: '',
        syllabusType: '',
        chunkCount: 0,
      },
    });
    uploadCourseSyllabusMock.mockRejectedValue(
      new ApiError('Only .pdf and .txt syllabus files are supported.', 400),
    );

    renderCreateCoursePage();
    fillRequiredTextFields();
    selectSyllabusFile();
    fireEvent.click(screen.getByRole('button', { name: 'Create course' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Only .pdf and .txt syllabus files are supported.',
    );
    expect(updateCourseMetadataMock).toHaveBeenCalledWith(
      'css-430-summer-2026-a82f',
      expect.objectContaining({ syllabusStatus: 'upload_failed' }),
    );
    expect(screen.getByTestId('location')).toHaveTextContent('/create-course');
  });

  it('displays Firebase save errors without uploading', async () => {
    createCourseMock.mockRejectedValue(new Error('Firebase permission denied'));

    renderCreateCoursePage();
    fillRequiredTextFields();
    selectSyllabusFile();
    fireEvent.click(screen.getByRole('button', { name: 'Create course' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Firebase permission denied',
    );
    expect(uploadCourseSyllabusMock).not.toHaveBeenCalled();
    expect(screen.getByTestId('location')).toHaveTextContent('/create-course');
  });
});
