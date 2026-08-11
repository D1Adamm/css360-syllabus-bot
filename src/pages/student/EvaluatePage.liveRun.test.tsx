/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

const addEvaluationMock = vi.fn();

vi.mock('../../hooks/useEvaluations', () => ({
  useEvaluations: () => ({
    evaluations: [],
    loading: false,
    error: null,
    saving: false,
    saveError: null,
    addEvaluation: addEvaluationMock,
    deleteEvaluation: vi.fn(),
    deleteAllEvaluations: vi.fn(),
    clearSaveError: vi.fn(),
  }),
}));

import { ComparisonRunProvider, type ComparisonRun } from '../../context/ComparisonRunContext';
import { CourseProvider } from '../../context/CourseContext';
import { EvaluatePage } from './EvaluatePage';

const COURSE_ID = 'css-360-winter-2026-a7rp';

function storeRun(run: Partial<ComparisonRun> = {}) {
  const full: ComparisonRun = {
    runId: 'run-1',
    courseId: COURSE_ID,
    question: 'How much of my grade is the final project?',
    matchedComparisonId: null,
    createdAt: '2026-01-01T00:00:00.000Z',
    responses: {
      base: { text: 'Base answer text', error: null, sources: [] },
      rag: { text: 'Syllabus-aware answer text', error: null, sources: ['Grading'] },
      fineTuned: { text: 'Course-trained answer text', error: null, sources: [] },
      fineTunedRag: { text: 'Combined answer text', error: null, sources: [] },
    },
    ...run,
  };
  window.sessionStorage.setItem(`sml.run.${COURSE_ID}`, JSON.stringify(full));
  return full;
}

function renderPage() {
  return render(
    <MemoryRouter initialEntries={[`/student/course/${COURSE_ID}/evaluate`]}>
      <Routes>
        <Route
          path="/student/course/:courseId/evaluate"
          element={
            <ComparisonRunProvider>
              <CourseProvider courseId={COURSE_ID}>
                <EvaluatePage />
              </CourseProvider>
            </ComparisonRunProvider>
          }
        />
      </Routes>
    </MemoryRouter>,
  );
}

function chooseAllCriteria(marker = 'Base Model') {
  for (const legend of [
    'Which answer was most accurate?',
    'Which was most helpful?',
    'Which was most concise?',
    'Which stayed closest to the syllabus?',
    'Which did you prefer overall?',
  ]) {
    const group = screen.getByRole('radiogroup', { name: legend });
    fireEvent.click(
      Array.from(group.querySelectorAll('label')).find((label) =>
        label.textContent?.includes(marker),
      )!,
    );
  }
}

beforeEach(() => {
  vi.clearAllMocks();
  window.sessionStorage.clear();
  addEvaluationMock.mockResolvedValue(undefined);
});

afterEach(() => {
  cleanup();
});

describe('EvaluatePage live run', () => {
  it('prompts the student to compare first when no run exists', () => {
    renderPage();

    expect(screen.getByText('Nothing to evaluate yet')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Go to Compare/ })).toHaveAttribute(
      'href',
      `/student/course/${COURSE_ID}/compare`,
    );
  });

  it('rates the exact responses generated on Compare, not bundled examples', () => {
    storeRun();
    renderPage();

    expect(screen.getByText('Base answer text')).toBeInTheDocument();
    expect(screen.getByText('Syllabus-aware answer text')).toBeInTheDocument();
    expect(screen.getByText('Course-trained answer text')).toBeInTheDocument();
    expect(screen.getByText('Combined answer text')).toBeInTheDocument();
    expect(
      screen.getByText('How much of my grade is the final project?'),
    ).toBeInTheDocument();
  });

  it('survives a reload by reading the run back from session storage', () => {
    storeRun();
    const first = renderPage();
    expect(first.getByText('Base answer text')).toBeInTheDocument();
    cleanup();

    // A fresh provider, as if the page had been reloaded.
    renderPage();
    expect(screen.getByText('Base answer text')).toBeInTheDocument();
  });

  it('requires every criterion before saving', async () => {
    storeRun();
    renderPage();

    fireEvent.click(screen.getByRole('button', { name: 'Submit evaluation' }));

    await waitFor(() => {
      expect(screen.getAllByText('Choose one answer.').length).toBe(5);
    });
    expect(addEvaluationMock).not.toHaveBeenCalled();
  });

  it('stores the run id and question wording alongside the ratings', async () => {
    storeRun();
    renderPage();

    chooseAllCriteria();
    fireEvent.click(screen.getByRole('button', { name: 'Submit evaluation' }));

    await waitFor(() => {
      expect(addEvaluationMock).toHaveBeenCalledTimes(1);
    });

    const saved = addEvaluationMock.mock.calls[0][0];
    expect(saved.runId).toBe('run-1');
    expect(saved.questionText).toBe('How much of my grade is the final project?');
    expect(saved.courseId).toBe(COURSE_ID);
    // Free-text questions get a synthetic id so aggregation still groups them.
    expect(saved.comparisonId).toBe('question-run-1');
    expect(saved.preferredModel).toBe('base');
  });

  it('keeps the predefined comparison id when the question matched an example', async () => {
    storeRun({ matchedComparisonId: 'comparison-2' });
    renderPage();

    chooseAllCriteria();
    fireEvent.click(screen.getByRole('button', { name: 'Submit evaluation' }));

    await waitFor(() => {
      expect(addEvaluationMock).toHaveBeenCalledTimes(1);
    });
    expect(addEvaluationMock.mock.calls[0][0].comparisonId).toBe('comparison-2');
  });

  it('offers a way back to Compare when nothing answered at all', () => {
    storeRun({
      responses: {
        base: { text: '', error: 'Temporarily unavailable.', sources: [] },
        rag: { text: '', error: 'Temporarily unavailable.', sources: [] },
        fineTuned: { text: '', error: 'Temporarily unavailable.', sources: [] },
        fineTunedRag: { text: '', error: 'Temporarily unavailable.', sources: [] },
      },
    });
    renderPage();

    expect(screen.getByText('No answers came through')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Ask again/ })).toHaveAttribute(
      'href',
      `/student/course/${COURSE_ID}/compare`,
    );
    // A form that cannot be completed is never shown.
    expect(
      screen.queryByRole('button', { name: 'Submit evaluation' }),
    ).not.toBeInTheDocument();
  });

  it('does not let an approach that failed be chosen as the best', () => {
    storeRun({
      responses: {
        base: { text: '', error: 'This response is temporarily unavailable.', sources: [] },
        rag: { text: 'Syllabus-aware answer text', error: null, sources: [] },
        fineTuned: { text: 'Course-trained answer text', error: null, sources: [] },
        fineTunedRag: { text: 'Combined answer text', error: null, sources: [] },
      },
    });
    renderPage();

    expect(screen.getByText('Some answers are unavailable')).toBeInTheDocument();

    const group = screen.getByRole('radiogroup', {
      name: 'Which answer was most accurate?',
    });
    const baseRadio = group.querySelector<HTMLInputElement>('input[value="base"]');
    expect(baseRadio).toBeDisabled();
  });

  it('confirms the rating was recorded', async () => {
    storeRun();
    renderPage();

    chooseAllCriteria();
    fireEvent.click(screen.getByRole('button', { name: 'Submit evaluation' }));

    expect(
      await screen.findByText('Thanks — your ratings were recorded'),
    ).toBeInTheDocument();
  });
});
