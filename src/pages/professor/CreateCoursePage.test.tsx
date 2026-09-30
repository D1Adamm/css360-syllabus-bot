/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { ApiError } from '../../lib/api';

const {
  createCourseMock,
  uploadCourseSyllabusMock,
  updateCourseMetadataMock,
  fetchSessionMock,
} = vi.hoisted(() => ({
  createCourseMock: vi.fn(),
  uploadCourseSyllabusMock: vi.fn(),
  updateCourseMetadataMock: vi.fn(),
  fetchSessionMock: vi.fn(),
}));

vi.mock('../../lib/authApi', async () => {
  const actual = await vi.importActual<typeof import('../../lib/authApi')>('../../lib/authApi');
  return { ...actual, fetchSession: fetchSessionMock };
});

vi.mock('../../lib/createCourse', () => ({
  createCourse: createCourseMock,
}));

vi.mock('../../lib/api', async () => {
  const actual = await vi.importActual<typeof import('../../lib/api')>('../../lib/api');
  return {
    ...actual,
    uploadCourseSyllabus: uploadCourseSyllabusMock,
  };
});

vi.mock('../../lib/coursesDb', async () => {
  const actual = await vi.importActual<typeof import('../../lib/coursesDb')>(
    '../../lib/coursesDb',
  );
  return {
    ...actual,
    updateCourseMetadata: updateCourseMetadataMock,
  };
});

import { RequireCourseStaff } from '../../components/auth/RouteGuards';
import { SessionProvider, type Session } from '../../context/SessionContext';
import { CreateCoursePage } from './CreateCoursePage';

const NEW_COURSE = 'css-430-summer-2026-a82f';

function professorSession(...courseIds: string[]): Session {
  return {
    user: {
      userId: 'user-prof',
      email: 'prof@uw.edu',
      displayName: 'Prof',
      role: 'professor',
      courseIds,
    },
    participants: [],
  };
}

function LocationProbe() {
  const location = useLocation();
  return <div data-testid="location">{location.pathname}</div>;
}

// A professor with no courses yet, and the real course-staff guard on the page
// the form navigates to: the route a professor lands on after creating a course
// is only reachable once the session knows about the new membership.
function renderCreateCoursePage(initialSession: Session = professorSession()) {
  return render(
    <MemoryRouter initialEntries={['/professor/courses/new']}>
      <SessionProvider initialSession={initialSession}>
        <Routes>
          <Route path="/professor/courses/new" element={<CreateCoursePage />} />
          <Route
            path="/professor/course/:courseId"
            element={
              <RequireCourseStaff>
                <div>Course home</div>
              </RequireCourseStaff>
            }
          />
          <Route path="/forbidden" element={<div>Forbidden</div>} />
        </Routes>
        <LocationProbe />
      </SessionProvider>
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
    // What /api/auth/session answers once the course exists: the creator now
    // holds its instructor membership.
    fetchSessionMock.mockResolvedValue(professorSession(NEW_COURSE));
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

  it('a professor with no courses creates one, uploads its syllabus and lands in it', async () => {
    createCourseMock.mockResolvedValue({
      courseId: NEW_COURSE,
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
      courseId: NEW_COURSE,
      syllabusFileName: 'css430-syllabus.pdf',
      syllabusType: 'pdf',
      syllabusStatus: 'indexed',
      fileSize: 16,
      characterCount: 18000,
      chunkCount: 18,
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
        NEW_COURSE,
        file,
      );
    });
    await waitFor(() => {
      expect(view.getByTestId('location')).toHaveTextContent(
        `/professor/course/${NEW_COURSE}`,
      );
    });
    // Regression: the session was re-read after creation, so the course-staff
    // guard admits the new instructor instead of redirecting to /forbidden.
    expect(fetchSessionMock).toHaveBeenCalled();
    expect(await screen.findByText('Course home')).toBeInTheDocument();
    expect(screen.queryByText('Forbidden')).not.toBeInTheDocument();
    // The upload records the syllabus on the course; the page writes nothing.
    expect(updateCourseMetadataMock).not.toHaveBeenCalled();
  });

  it('a failed upload keeps the created course and a retry uploads into it', async () => {
    createCourseMock.mockResolvedValue({
      courseId: NEW_COURSE,
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
    uploadCourseSyllabusMock
      .mockRejectedValueOnce(new ApiError('Ollama is unavailable for embeddings.', 503))
      .mockResolvedValueOnce({
        courseId: NEW_COURSE,
        syllabusFileName: 'css430-syllabus.pdf',
        syllabusType: 'pdf',
        syllabusStatus: 'indexed',
        fileSize: 16,
        characterCount: 18000,
        chunkCount: 18,
      });

    renderCreateCoursePage();
    fillRequiredTextFields();
    const file = selectSyllabusFile();
    fireEvent.click(screen.getByRole('button', { name: 'Create course' }));

    // It says the course exists, offers a way into it, and stays put.
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(/course was created, but its syllabus could not be processed/i);
    expect(alert).not.toHaveTextContent(/ollama/i);
    expect(within(alert).getByRole('link', { name: 'Open the course' })).toHaveAttribute(
      'href',
      `/professor/course/${NEW_COURSE}/syllabus`,
    );
    expect(screen.getByTestId('location')).toHaveTextContent('/professor/courses/new');
    // The course details belong to the created course now and are locked.
    expect(screen.getByLabelText(/Course name or code/)).toBeDisabled();

    // Retrying uploads into the same course; nothing is created twice.
    fireEvent.click(screen.getByRole('button', { name: 'Upload syllabus' }));
    await waitFor(() => {
      expect(screen.getByTestId('location')).toHaveTextContent(
        `/professor/course/${NEW_COURSE}`,
      );
    });
    expect(createCourseMock).toHaveBeenCalledTimes(1);
    expect(uploadCourseSyllabusMock).toHaveBeenCalledTimes(2);
    expect(uploadCourseSyllabusMock).toHaveBeenLastCalledWith(NEW_COURSE, file);
    expect(updateCourseMetadataMock).not.toHaveBeenCalled();
  });

  it('refuses a file over 10 MB before creating anything', async () => {
    renderCreateCoursePage();
    fillRequiredTextFields();
    const big = new File(['x'], 'syllabus.pdf', { type: 'application/pdf' });
    Object.defineProperty(big, 'size', { value: 10 * 1024 * 1024 + 1 });
    fireEvent.change(screen.getByLabelText(/Syllabus file/), { target: { files: [big] } });
    fireEvent.click(screen.getByRole('button', { name: 'Create course' }));

    expect(await screen.findByText('Syllabus files must be 10 MB or smaller.')).toBeInTheDocument();
    expect(createCourseMock).not.toHaveBeenCalled();
  });

  it('reports a save failure without naming the database', async () => {
    createCourseMock.mockRejectedValue(new Error('PostgreSQL permission denied'));

    renderCreateCoursePage();
    fillRequiredTextFields();
    selectSyllabusFile();
    fireEvent.click(screen.getByRole('button', { name: 'Create course' }));

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(/try uploading it again|try again/i);
    expect(alert).not.toHaveTextContent(/postgres/i);
    expect(uploadCourseSyllabusMock).not.toHaveBeenCalled();
    expect(screen.getByTestId('location')).toHaveTextContent(
      '/professor/courses/new',
    );
  });
});
