/** @vitest-environment jsdom */
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { ApiError } from '../lib/api';
import { CourseProvider } from '../context/CourseContext';
import { SyllabusPage } from './SyllabusPage';

const fetchCourseSyllabusTextMock = vi.hoisted(() => vi.fn());

vi.mock('../lib/api', async () => {
  const actual = await vi.importActual<typeof import('../lib/api')>('../lib/api');
  return {
    ...actual,
    fetchCourseSyllabusText: fetchCourseSyllabusTextMock,
  };
});

function renderSyllabusPage(courseId: string) {
  return render(
    <MemoryRouter initialEntries={[`/course/${courseId}/syllabus`]}>
      <Routes>
        <Route
          path="/course/:courseId/syllabus"
          element={
            <CourseProvider courseId={courseId}>
              <SyllabusPage />
            </CourseProvider>
          }
        />
      </Routes>
    </MemoryRouter>,
  );
}

describe('SyllabusPage course-specific text', () => {
  beforeEach(() => {
    fetchCourseSyllabusTextMock.mockReset();
  });

  afterEach(() => {
    cleanup();
  });

  it('requests CSS 350 syllabus text for the CSS 350 route', async () => {
    fetchCourseSyllabusTextMock.mockResolvedValue({
      courseId: 'css-350-spring-2026-abcd',
      text: 'CSS 350 Course Policies\n\nAttendance is required for CSS 350.',
      characterCount: 60,
    });

    renderSyllabusPage('css-350-spring-2026-abcd');

    await waitFor(() => {
      expect(fetchCourseSyllabusTextMock).toHaveBeenCalledWith('css-350-spring-2026-abcd');
    });
    expect(await screen.findByText('Attendance is required for CSS 350.')).toBeInTheDocument();
  });

  it('requests CSS 430 syllabus text for the CSS 430 route', async () => {
    fetchCourseSyllabusTextMock.mockResolvedValue({
      courseId: 'css-430-summer-2026-ibce',
      text: 'CSS 430 Late Policy\n\nCSS 430 late work may receive half credit.',
      characterCount: 68,
    });

    renderSyllabusPage('css-430-summer-2026-ibce');

    await waitFor(() => {
      expect(fetchCourseSyllabusTextMock).toHaveBeenCalledWith('css-430-summer-2026-ibce');
    });
    expect(
      await screen.findByText('CSS 430 late work may receive half credit.'),
    ).toBeInTheDocument();
  });

  it('displays different syllabus content for two courses', async () => {
    fetchCourseSyllabusTextMock.mockImplementation(async (courseId: string) => {
      if (courseId === 'css-350-spring-2026-abcd') {
        return {
          courseId,
          text: 'CSS 350 unique syllabus content about networks.',
          characterCount: 47,
        };
      }

      return {
        courseId,
        text: 'CSS 430 unique syllabus content about processes.',
        characterCount: 48,
      };
    });

    const first = renderSyllabusPage('css-350-spring-2026-abcd');
    expect(
      await screen.findByText('CSS 350 unique syllabus content about networks.'),
    ).toBeInTheDocument();
    first.unmount();

    renderSyllabusPage('css-430-summer-2026-ibce');
    expect(
      await screen.findByText('CSS 430 unique syllabus content about processes.'),
    ).toBeInTheDocument();
    expect(
      screen.queryByText('CSS 350 unique syllabus content about networks.'),
    ).not.toBeInTheDocument();
  });

  it('shows a loading state while the syllabus request is in flight', async () => {
    let resolveRequest: ((value: {
      courseId: string;
      text: string;
      characterCount: number;
    }) => void) | undefined;

    fetchCourseSyllabusTextMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveRequest = resolve;
        }),
    );

    renderSyllabusPage('css-430-summer-2026-ibce');
    expect(screen.getByRole('heading', { name: 'Loading syllabus' })).toBeInTheDocument();

    resolveRequest?.({
      courseId: 'css-430-summer-2026-ibce',
      text: 'Loaded syllabus body.',
      characterCount: 21,
    });

    expect(await screen.findByText('Loaded syllabus body.')).toBeInTheDocument();
  });

  it('shows a missing syllabus state', async () => {
    fetchCourseSyllabusTextMock.mockRejectedValue(
      new ApiError('Extracted syllabus text was not found for this course.', 404),
    );

    renderSyllabusPage('course-without-syllabus');
    expect(await screen.findByRole('heading', { name: 'Syllabus not found' })).toBeInTheDocument();
    expect(screen.getByText(/No extracted syllabus text is available/i)).toBeInTheDocument();
  });

  it('shows a backend unavailable state', async () => {
    fetchCourseSyllabusTextMock.mockRejectedValue(
      new ApiError(
        'Could not reach the backend to load the syllabus. Make sure the FastAPI server is running.',
      ),
    );

    renderSyllabusPage('css-430-summer-2026-ibce');
    expect(await screen.findByRole('heading', { name: 'Backend unavailable' })).toBeInTheDocument();
  });

  it('shows an invalid course state from the API', async () => {
    fetchCourseSyllabusTextMock.mockRejectedValue(
      new ApiError('Invalid courseId "Bad_Id": must be non-empty...', 400),
    );

    renderSyllabusPage('css-430-summer-2026-ibce');
    expect(await screen.findByRole('heading', { name: 'Invalid course' })).toBeInTheDocument();
  });

  it('preserves line breaks in the rendered syllabus document', async () => {
    fetchCourseSyllabusTextMock.mockResolvedValue({
      courseId: 'css-430-summer-2026-ibce',
      text: 'Office Hours\n\nMondays 1-2pm\nWednesdays 3-4pm',
      characterCount: 44,
    });

    renderSyllabusPage('css-430-summer-2026-ibce');
    const documentNode = await screen.findByTestId('syllabus-document');
    expect(documentNode.querySelector('h2')).toHaveTextContent('Office Hours');
    const paragraph = documentNode.querySelector('p');
    expect(paragraph?.innerHTML).toContain('<br');
    expect(paragraph).toHaveTextContent(/Mondays 1-2pm/);
    expect(paragraph).toHaveTextContent(/Wednesdays 3-4pm/);
  });
});
