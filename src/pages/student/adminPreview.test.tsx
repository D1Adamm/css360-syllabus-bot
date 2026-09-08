/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

/**
 * An administrator's preview of the student pages.
 *
 * Administrators never redeem a classroom code, so on a course's student pages
 * there is no participant to attribute a rating to. The pages run exactly as
 * they do for a student — the same four generation requests for the course in
 * the URL — and the last step changes: Evaluate submits to the backend's
 * preview route, which validates and stores nothing, and Contribute submits
 * nowhere. A participant, and a professor, get the unchanged behaviour.
 *
 * The banner the shell shows in preview, and the button on the admin course
 * page that starts one, are covered in App.routes.test.tsx, where the real
 * shell and route guards render.
 */

const COURSE = vi.hoisted(() => 'css-360-winter-2026-a7rp');

const evaluationsDb = vi.hoisted(() => ({
  createEvaluation: vi.fn(),
  previewEvaluation: vi.fn(),
  subscribeToEvaluations: vi.fn(),
}));

vi.mock('../../lib/evaluationsDb', () => ({
  createEvaluation: evaluationsDb.createEvaluation,
  previewEvaluation: evaluationsDb.previewEvaluation,
  subscribeToEvaluations: evaluationsDb.subscribeToEvaluations,
  deleteEvaluation: vi.fn(),
  deleteAllEvaluations: vi.fn(),
}));

const api = vi.hoisted(() => ({
  generateBaseModel: vi.fn(),
  generateRag: vi.fn(),
  generateFineTuned: vi.fn(),
  generateFineTunedRag: vi.fn(),
}));

vi.mock('../../lib/api', () => ({
  ApiError: class ApiError extends Error {},
  ...api,
  listCourseSeeds: vi.fn(async () => ({ courseId: COURSE, count: 0, seeds: [] })),
}));

const addSeedMock = vi.hoisted(() => vi.fn());

vi.mock('../../hooks/useSeedExamples', () => ({
  useSeedExamples: () => ({
    seeds: [],
    loading: false,
    error: null,
    saving: false,
    saveError: null,
    addSeed: addSeedMock,
    deleteSeed: vi.fn(),
    deleteAllSeeds: vi.fn(),
    clearSaveError: vi.fn(),
  }),
}));

import { CourseRoute } from '../../components/CourseRoute';
import {
  ComparisonRunProvider,
  type ComparisonRun,
} from '../../context/ComparisonRunContext';
import { SessionProvider, type Session } from '../../context/SessionContext';
import { ComparePage } from './ComparePage';
import { ContributePage } from './ContributePage';
import { EvaluatePage } from './EvaluatePage';

/** `admin` has joined nothing; `student` joined the course; `professor` teaches it. */
const SESSIONS: Record<'admin' | 'student' | 'professor', Session> = {
  admin: {
    user: {
      userId: 'u-admin',
      email: 'admin@uw.edu',
      displayName: 'Admin Example',
      role: 'admin',
      courseIds: [],
    },
    participant: null,
  },
  student: { user: null, participant: { courseId: COURSE } },
  professor: {
    user: {
      userId: 'u-prof',
      email: 'prof@uw.edu',
      displayName: 'Prof Example',
      role: 'professor',
      courseIds: [COURSE],
    },
    participant: null,
  },
};

type Who = keyof typeof SESSIONS;

const PREVIEW_NOTE = 'Preview mode — responses are not saved';

function renderStudentPages(path: string, who: Who) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <SessionProvider initialSession={SESSIONS[who]}>
        <ComparisonRunProvider>
          <Routes>
            <Route path="/student/course/:courseId" element={<CourseRoute adminPreview />}>
              <Route path="compare" element={<ComparePage />} />
              <Route path="evaluate" element={<EvaluatePage />} />
              <Route path="contribute" element={<ContributePage />} />
            </Route>
          </Routes>
        </ComparisonRunProvider>
      </SessionProvider>
    </MemoryRouter>,
  );
}

function storeRun() {
  const run: ComparisonRun = {
    runId: 'run-1',
    courseId: COURSE,
    question: 'How much of my grade is the final project?',
    matchedComparisonId: null,
    createdAt: '2026-01-01T00:00:00.000Z',
    responses: {
      base: { text: 'Base answer text', error: null, sources: [] },
      rag: { text: 'RAG answer text', error: null, sources: ['Grading'] },
      fineTuned: { text: 'Fine-tuned answer text', error: null, sources: [] },
      fineTunedRag: { text: 'Combined answer text', error: null, sources: [] },
    },
  };
  window.sessionStorage.setItem(`sml.run.${COURSE}`, JSON.stringify(run));
}

function chooseAllCriteria() {
  for (const legend of [
    'Which answer was most accurate?',
    'Which answer would you prefer overall?',
  ]) {
    const group = screen.getByRole('radiogroup', { name: legend });
    fireEvent.click(
      Array.from(group.querySelectorAll('label')).find((label) =>
        label.textContent?.includes('RAG'),
      )!,
    );
  }
}

function submitRating() {
  chooseAllCriteria();
  fireEvent.click(screen.getByRole('button', { name: 'Submit evaluation' }));
}

function fillContribution() {
  fireEvent.change(screen.getByLabelText(/Your question/), {
    target: { value: 'How much of my grade comes from the final project?' },
  });
  fireEvent.change(screen.getByLabelText(/The answer you would expect/), {
    target: { value: 'The final project is worth 30% of the overall course grade.' },
  });
  fireEvent.click(screen.getByRole('button', { name: /Add question/ }));
}

beforeEach(() => {
  vi.clearAllMocks();
  window.sessionStorage.clear();

  evaluationsDb.subscribeToEvaluations.mockImplementation(
    (_courseId: string, onData: (value: unknown[]) => void) => {
      onData([]);
      return () => {};
    },
  );
  const record = { id: 'x', courseId: COURSE };
  evaluationsDb.createEvaluation.mockResolvedValue(record);
  evaluationsDb.previewEvaluation.mockResolvedValue(record);
  addSeedMock.mockResolvedValue(undefined);

  api.generateBaseModel.mockResolvedValue({
    answer: 'Base answer',
    model: 'm',
    responseType: 'base',
  });
  api.generateRag.mockResolvedValue({
    courseId: COURSE,
    answer: 'RAG answer',
    model: 'm',
    responseType: 'rag',
    sources: [],
    retrievedChunks: [],
  });
  api.generateFineTuned.mockResolvedValue({
    answer: 'Fine-tuned answer',
    model: 'm',
    responseType: 'fineTuned',
    adapterLoaded: true,
  });
  api.generateFineTunedRag.mockResolvedValue({
    courseId: COURSE,
    answer: 'Combined answer',
    model: 'm',
    responseType: 'fineTunedRag',
    adapterLoaded: true,
    sources: [],
    retrievedChunks: [],
  });
});

afterEach(() => {
  cleanup();
});

describe('Evaluate in an administrator preview', () => {
  it('submits the rating to the preview route for the course in the URL, and stores nothing', async () => {
    storeRun();
    renderStudentPages(`/student/course/${COURSE}/evaluate`, 'admin');

    expect(screen.getByText(PREVIEW_NOTE)).toBeInTheDocument();
    // The form itself is the student's form.
    expect(screen.getByRole('button', { name: 'Submit evaluation' })).toBeInTheDocument();

    submitRating();

    await waitFor(() => {
      expect(evaluationsDb.previewEvaluation).toHaveBeenCalledTimes(1);
    });
    const [courseId, rating] = evaluationsDb.previewEvaluation.mock.calls[0];
    expect(courseId).toBe(COURSE);
    expect(rating.courseId).toBe(COURSE);
    expect(rating.preferredModel).toBe('rag');
    expect(evaluationsDb.createEvaluation).not.toHaveBeenCalled();

    expect(
      await screen.findByText('Preview complete — nothing was saved'),
    ).toBeInTheDocument();
  });

  it("records a participant's rating exactly as before", async () => {
    storeRun();
    renderStudentPages(`/student/course/${COURSE}/evaluate`, 'student');

    expect(screen.queryByText(PREVIEW_NOTE)).toBeNull();
    submitRating();

    await waitFor(() => {
      expect(evaluationsDb.createEvaluation).toHaveBeenCalledTimes(1);
    });
    expect(evaluationsDb.createEvaluation.mock.calls[0][0]).toBe(COURSE);
    expect(evaluationsDb.previewEvaluation).not.toHaveBeenCalled();
    expect(
      await screen.findByText('Thanks — your ratings were recorded'),
    ).toBeInTheDocument();
  });

  it('leaves a professor who has not joined the course on the real route, as before', async () => {
    // Unchanged on purpose. A professor previews through their own classroom
    // code, as a real participant; without one the backend refuses the
    // rating, exactly as it did before administrator preview existed.
    storeRun();
    renderStudentPages(`/student/course/${COURSE}/evaluate`, 'professor');

    expect(screen.queryByText(PREVIEW_NOTE)).toBeNull();
    submitRating();

    await waitFor(() => {
      expect(evaluationsDb.createEvaluation).toHaveBeenCalledTimes(1);
    });
    expect(evaluationsDb.previewEvaluation).not.toHaveBeenCalled();
  });
});

describe('Compare in an administrator preview', () => {
  it('asks the same four routes for the course in the URL', async () => {
    renderStudentPages(`/student/course/${COURSE}/compare`, 'admin');

    fireEvent.change(
      screen.getByLabelText('What would you like to ask about this course?'),
      { target: { value: 'What is the late policy?' } },
    );
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }));

    await waitFor(() => {
      expect(api.generateBaseModel).toHaveBeenCalled();
      expect(api.generateRag).toHaveBeenCalled();
      expect(api.generateFineTuned).toHaveBeenCalled();
      expect(api.generateFineTunedRag).toHaveBeenCalled();
    });
    for (const generate of Object.values(api)) {
      expect(generate.mock.calls[0][0]).toBe(COURSE);
      expect(generate.mock.calls[0][1]).toBe('What is the late policy?');
    }

    expect(await screen.findByText('Base answer')).toBeInTheDocument();
    expect(
      await screen.findByRole('link', { name: /Evaluate these responses/ }),
    ).toHaveAttribute('href', `/student/course/${COURSE}/evaluate`);
  });
});

describe('Contribute in an administrator preview', () => {
  it('shows the student form and submits nowhere', async () => {
    renderStudentPages(`/student/course/${COURSE}/contribute`, 'admin');

    fillContribution();

    expect(
      await screen.findByText('Preview mode — contributions are not saved'),
    ).toBeInTheDocument();
    expect(addSeedMock).not.toHaveBeenCalled();
  });

  it("still adds a participant's question", async () => {
    renderStudentPages(`/student/course/${COURSE}/contribute`, 'student');

    fillContribution();

    await waitFor(() => {
      expect(addSeedMock).toHaveBeenCalledTimes(1);
    });
    expect(screen.queryByText('Preview mode — contributions are not saved')).toBeNull();
  });
});
