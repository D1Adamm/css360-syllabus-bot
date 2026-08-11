/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Link, MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

/**
 * Cross-course isolation for the student comparison flow.
 *
 * The backend keys every retrieval and syllabus read on the course id, and
 * these tests hold the frontend to the same standard: the id in the URL is the
 * only thing that decides which course's data is requested or displayed, and
 * nothing from one course may survive a switch to another.
 */

vi.mock('../../lib/api', async () => {
  const actual = await vi.importActual<typeof import('../../lib/api')>('../../lib/api');
  return {
    ...actual,
    generateBaseModel: vi.fn(),
    generateRag: vi.fn(),
    generateFineTuned: vi.fn(),
    generateFineTunedRag: vi.fn(),
    listCourseSeeds: vi.fn(),
  };
});

import {
  generateBaseModel,
  generateFineTuned,
  generateFineTunedRag,
  generateRag,
  listCourseSeeds,
} from '../../lib/api';
import { ComparisonRunProvider } from '../../context/ComparisonRunContext';
import { CourseRoute } from '../../components/CourseRoute';
import { ComparePage } from './ComparePage';
import { EvaluatePage } from './EvaluatePage';

const generateBaseModelMock = vi.mocked(generateBaseModel);
const generateRagMock = vi.mocked(generateRag);
const generateFineTunedMock = vi.mocked(generateFineTuned);
const generateFineTunedRagMock = vi.mocked(generateFineTunedRag);
const listCourseSeedsMock = vi.mocked(listCourseSeeds);

const COURSE_A = 'css-360-winter-2026-a7rp';
const COURSE_B = 'css-490-spring-2026-cgvl';

function mockAnswers(label: string) {
  generateBaseModelMock.mockResolvedValue({
    answer: `${label} base answer`,
    model: 'm',
    responseType: 'base',
  });
  generateRagMock.mockResolvedValue({
    courseId: 'ignored',
    answer: `${label} rag answer`,
    model: 'm',
    responseType: 'rag',
    sources: [],
    retrievedChunks: [],
  });
  generateFineTunedMock.mockResolvedValue({
    answer: `${label} fine-tuned answer`,
    model: 'm',
    responseType: 'fineTuned',
    adapterLoaded: true,
  });
  generateFineTunedRagMock.mockResolvedValue({
    courseId: 'ignored',
    answer: `${label} combined answer`,
    model: 'm',
    responseType: 'fineTunedRag',
    adapterLoaded: true,
    sources: [],
    retrievedChunks: [],
  });
}

/**
 * Renders the real course route tree so the courseId comes from the URL.
 *
 * A permanent link to the other course lets a test perform a genuine in-app
 * navigation between two courses that share the same route pattern — which is
 * exactly the case where React Router reuses component instances.
 */
function renderAt(path: string, switchTo?: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <ComparisonRunProvider>
        {switchTo && <Link to={switchTo}>Switch course</Link>}
        <Routes>
          <Route path="/student/course/:courseId" element={<CourseRoute />}>
            <Route path="compare" element={<ComparePage />} />
            <Route path="evaluate" element={<EvaluatePage />} />
          </Route>
        </Routes>
      </ComparisonRunProvider>
    </MemoryRouter>,
  );
}

function askQuestion(question: string) {
  fireEvent.change(
    screen.getByLabelText('What would you like to ask about this course?'),
    { target: { value: question } },
  );
  fireEvent.click(screen.getByRole('button', { name: /^(Ask|Asking…)$/ }));
}

beforeEach(() => {
  vi.clearAllMocks();
  window.sessionStorage.clear();
  mockAnswers('course A');
  listCourseSeedsMock.mockResolvedValue({
    courseId: COURSE_A,
    count: 0,
    firebasePath: '',
    seeds: [],
  });
});

afterEach(() => {
  cleanup();
});

describe('comparison requests are scoped to the route course', () => {
  it('sends the URL course id to every one of the four model calls', async () => {
    renderAt(`/student/course/${COURSE_B}/compare`);
    askQuestion('What is the late work policy?');

    await waitFor(() => {
      expect(generateRagMock).toHaveBeenCalled();
    });

    for (const mock of [
      generateBaseModelMock,
      generateRagMock,
      generateFineTunedMock,
      generateFineTunedRagMock,
    ]) {
      expect(mock.mock.calls[0]?.[0]).toBe(COURSE_B);
    }
  });

  it('asks for suggestions from the route course, not a global list', async () => {
    renderAt(`/student/course/${COURSE_B}/compare`);

    await waitFor(() => {
      expect(listCourseSeedsMock).toHaveBeenCalledWith(COURSE_B);
    });
  });
});

describe('no comparison state leaks between courses', () => {
  it('clears a previous course’s answers when the course id changes', async () => {
    renderAt(`/student/course/${COURSE_A}/compare`, `/student/course/${COURSE_B}/compare`);
    askQuestion('What is the late work policy?');

    expect(await screen.findByText('course A base answer')).toBeInTheDocument();

    // Same route pattern, different course — React Router reuses component
    // instances here, so the page must not carry the old answers across.
    fireEvent.click(screen.getByRole('link', { name: 'Switch course' }));

    await waitFor(() => {
      expect(screen.queryByText('course A base answer')).not.toBeInTheDocument();
    });
    expect(screen.queryByText('course A rag answer')).not.toBeInTheDocument();
    expect(screen.getByText('No question asked yet')).toBeInTheDocument();
  });

  it('keeps each course’s stored run separate', async () => {
    renderAt(`/student/course/${COURSE_A}/compare`);
    askQuestion('Course A question');
    await screen.findByText('course A base answer');
    cleanup();

    mockAnswers('course B');
    renderAt(`/student/course/${COURSE_B}/compare`);
    askQuestion('Course B question');
    await screen.findByText('course B base answer');
    cleanup();

    const runA = JSON.parse(String(window.sessionStorage.getItem(`sml.run.${COURSE_A}`)));
    const runB = JSON.parse(String(window.sessionStorage.getItem(`sml.run.${COURSE_B}`)));

    expect(runA.courseId).toBe(COURSE_A);
    expect(runA.question).toBe('Course A question');
    expect(runA.responses.base.text).toBe('course A base answer');

    expect(runB.courseId).toBe(COURSE_B);
    expect(runB.question).toBe('Course B question');
    expect(runB.responses.base.text).toBe('course B base answer');
  });

  it('never offers one course’s run for evaluation under another course', async () => {
    renderAt(`/student/course/${COURSE_A}/compare`);
    askQuestion('Course A question');
    await screen.findByText('course A base answer');
    cleanup();

    // Course B has no run of its own, so Evaluate must be empty — not showing
    // course A's answers.
    renderAt(`/student/course/${COURSE_B}/evaluate`);

    expect(screen.getByText('Nothing to evaluate yet')).toBeInTheDocument();
    expect(screen.queryByText('course A base answer')).not.toBeInTheDocument();
    expect(screen.queryByText('Course A question')).not.toBeInTheDocument();
  });
});
