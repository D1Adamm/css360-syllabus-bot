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
import { ApiError } from '../../lib/api';
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
      rag: { text: 'RAG answer text', error: null, sources: ['Grading'] },
      fineTuned: { text: 'Fine-tuned answer text', error: null, sources: [] },
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

/** The two single-choice criteria the form still asks for. */
const CRITERION_LEGENDS = [
  'Which answer was most accurate?',
  'Which answer would you prefer overall?',
];

/** Criteria the form used to ask and no longer does. */
const RETIRED_LEGENDS = [
  'Which was most helpful?',
  'Which was most concise?',
  'Which stayed closest to the syllabus?',
];

function chooseAllCriteria(marker = 'Base') {
  for (const legend of CRITERION_LEGENDS) {
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
    expect(screen.getByText('RAG answer text')).toBeInTheDocument();
    expect(screen.getByText('Fine-tuned answer text')).toBeInTheDocument();
    expect(screen.getByText('Combined answer text')).toBeInTheDocument();
    expect(
      screen.getByText('How much of my grade is the final project?'),
    ).toBeInTheDocument();
  });

  it('names the four approaches by their technical approach', () => {
    storeRun();
    renderPage();

    const group = screen.getByRole('radiogroup', {
      name: 'Which answer was most accurate?',
    });
    const options = Array.from(group.querySelectorAll('.criterion__option-label')).map(
      (option) => option.textContent,
    );

    expect(options).toEqual(['Base', 'RAG', 'Fine-Tuned', 'Fine-Tuned + RAG']);
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
      expect(screen.getAllByText('Choose one answer.').length).toBe(
        CRITERION_LEGENDS.length,
      );
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
        rag: { text: 'RAG answer text', error: null, sources: [] },
        fineTuned: { text: 'Fine-tuned answer text', error: null, sources: [] },
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
  /* --------------------------------------------------------------------- *
   * The simplified form
   *
   * Five single-choice criteria over the same four answers was enough friction
   * that ratings stopped getting finished. Three of them — helpfulness,
   * concision, closeness to the syllabus — were retired. The data they produced
   * is untouched; the questions are simply no longer asked.
   * --------------------------------------------------------------------- */

  it('asks only the two single-choice criteria that remain', () => {
    storeRun();
    renderPage();

    for (const legend of CRITERION_LEGENDS) {
      expect(screen.getByRole('radiogroup', { name: legend })).toBeInTheDocument();
    }
    expect(screen.getAllByRole('radiogroup')).toHaveLength(CRITERION_LEGENDS.length);
  });

  it('no longer asks the retired criteria', () => {
    storeRun();
    renderPage();

    for (const legend of RETIRED_LEGENDS) {
      expect(screen.queryByRole('radiogroup', { name: legend })).not.toBeInTheDocument();
    }
    expect(document.body.textContent ?? '').not.toMatch(/most helpful|most concise/i);
  });

  it('still asks which answers were unsupported, and still takes a comment', () => {
    storeRun();
    renderPage();

    expect(
      screen.getByText('Did any answer include information the syllabus does not support?'),
    ).toBeInTheDocument();
    const flagged = screen
      .getAllByRole('checkbox')
      .map((box) => box.closest('label')?.textContent?.replace(/^[A-D]/, '') ?? '');
    expect(flagged).toEqual(['Base', 'RAG', 'Fine-Tuned', 'Fine-Tuned + RAG']);
    expect(screen.getByPlaceholderText(/What made one answer better/)).toBeInTheDocument();
  });

  it('omits the retired criteria from a new record rather than inventing answers', async () => {
    storeRun();
    renderPage();

    chooseAllCriteria();
    fireEvent.click(screen.getByRole('button', { name: 'Submit evaluation' }));

    await waitFor(() => {
      expect(addEvaluationMock).toHaveBeenCalledTimes(1);
    });

    const saved = addEvaluationMock.mock.calls[0][0];
    expect(saved.mostAccurate).toBe('base');
    expect(saved.preferredModel).toBe('base');
    expect(saved.mostHelpful).toBeUndefined();
    expect(saved.mostConcise).toBeUndefined();
    expect(saved.bestGrounded).toBeUndefined();
  });

  it('records the approaches a student flagged as unsupported', async () => {
    storeRun();
    renderPage();

    chooseAllCriteria();
    fireEvent.click(screen.getAllByRole('checkbox')[0]);
    fireEvent.click(screen.getByRole('button', { name: 'Submit evaluation' }));

    await waitFor(() => {
      expect(addEvaluationMock).toHaveBeenCalledTimes(1);
    });
    expect(addEvaluationMock.mock.calls[0][0].hallucinationFlags).toEqual(['base']);
  });

  /* --------------------------------------------------------------------- *
   * What a rating records, and rating once
   *
   * Audit findings B8 and B9 (fixed 2026-11-03). A rating used to store a
   * winner and a run id, but not the answers it was choosing between or the
   * model version that produced them; and nothing stopped the same run being
   * rated twice. The record now carries a snapshot of the four answers, the
   * stored run is marked rated, and the backend's refusal of a second rating
   * is treated as "already rated" rather than as a failure.
   * --------------------------------------------------------------------- */

  it('stores a snapshot of the four answers, with the model and version that produced each', async () => {
    storeRun({
      responses: {
        base: { text: 'Base answer text', error: null, sources: [], model: 'llama3.2:3b' },
        rag: { text: 'RAG answer text', error: null, sources: ['Grading'], model: 'llama3.2:3b' },
        fineTuned: {
          text: 'Fine-tuned answer text',
          error: null,
          sources: [],
          model: 'css-360-ft',
          modelVersion: 'v2',
          engine: 'vm',
        },
        fineTunedRag: { text: '', error: 'This response is temporarily unavailable.', sources: [] },
      },
    });
    renderPage();

    chooseAllCriteria();
    fireEvent.click(screen.getByRole('button', { name: 'Submit evaluation' }));

    await waitFor(() => {
      expect(addEvaluationMock).toHaveBeenCalledTimes(1);
    });
    const saved = addEvaluationMock.mock.calls[0][0];
    expect(saved.responses).toEqual({
      base: {
        text: 'Base answer text',
        error: null,
        sources: [],
        model: 'llama3.2:3b',
        modelVersion: null,
        engine: null,
      },
      rag: {
        text: 'RAG answer text',
        error: null,
        sources: ['Grading'],
        model: 'llama3.2:3b',
        modelVersion: null,
        engine: null,
      },
      fineTuned: {
        text: 'Fine-tuned answer text',
        error: null,
        sources: [],
        model: 'css-360-ft',
        modelVersion: 'v2',
        engine: 'vm',
      },
      fineTunedRag: {
        text: '',
        error: 'This response is temporarily unavailable.',
        sources: [],
        model: null,
        modelVersion: null,
        engine: null,
      },
    });
  });

  it('marks the run rated once saved, so coming back shows the thanks and not the form', async () => {
    storeRun();
    addEvaluationMock.mockResolvedValue({ id: 'evaluation-stored-1' });
    renderPage();

    chooseAllCriteria();
    fireEvent.click(screen.getByRole('button', { name: 'Submit evaluation' }));
    expect(await screen.findByText('Thanks — your ratings were recorded')).toBeInTheDocument();

    const stored = JSON.parse(window.sessionStorage.getItem(`sml.run.${COURSE_ID}`) ?? 'null');
    expect(stored.ratedAt).toEqual(expect.any(String));
    expect(stored.evaluationId).toBe('evaluation-stored-1');

    // The page is opened again (a reload, or the home page's link).
    cleanup();
    renderPage();
    expect(screen.getByText('You already rated this comparison')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Submit evaluation' })).not.toBeInTheDocument();
    expect(addEvaluationMock).toHaveBeenCalledTimes(1);
  });

  it("treats the backend's refusal of a second rating as already rated, not as an error", async () => {
    storeRun();
    addEvaluationMock.mockRejectedValue(
      new ApiError('You have already rated this comparison.', 409, 'evaluation_already_recorded'),
    );
    renderPage();

    chooseAllCriteria();
    fireEvent.click(screen.getByRole('button', { name: 'Submit evaluation' }));

    expect(await screen.findByText('You already rated this comparison')).toBeInTheDocument();
    expect(screen.queryByText('Not saved')).not.toBeInTheDocument();
    const stored = JSON.parse(window.sessionStorage.getItem(`sml.run.${COURSE_ID}`) ?? 'null');
    expect(stored.ratedAt).toEqual(expect.any(String));
    expect(stored.evaluationId).toBeNull();
  });

  it('does not mark a newer run rated when a new comparison settled while the rating was saving', async () => {
    // Comparisons keep running after the student leaves Compare. If Q2 settles
    // during the POST for Q1's rating, the provider has already replaced the
    // stored run; marking it would lock the student out of rating Q2.
    storeRun();
    let resolveSave: (value: unknown) => void = () => undefined;
    addEvaluationMock.mockImplementation(
      () => new Promise((resolve) => { resolveSave = resolve; }),
    );
    renderPage();

    chooseAllCriteria();
    fireEvent.click(screen.getByRole('button', { name: 'Submit evaluation' }));
    await waitFor(() => {
      expect(addEvaluationMock).toHaveBeenCalledTimes(1);
    });

    // Q2 settles and is saved by the provider while the request is in flight.
    const newer = storeRun({ runId: 'run-2', question: 'Is there a final exam?' });
    resolveSave({ id: 'evaluation-stored-1' });

    expect(await screen.findByText('Thanks — your ratings were recorded')).toBeInTheDocument();
    const stored = JSON.parse(window.sessionStorage.getItem(`sml.run.${COURSE_ID}`) ?? 'null');
    expect(stored.runId).toBe(newer.runId);
    expect(stored.ratedAt).toBeUndefined();
  });

  it('keeps the rated run when the student goes back to compare another question', async () => {
    storeRun();
    addEvaluationMock.mockResolvedValue({ id: 'evaluation-stored-1' });
    renderPage();

    chooseAllCriteria();
    fireEvent.click(screen.getByRole('button', { name: 'Submit evaluation' }));
    fireEvent.click(await screen.findByRole('link', { name: 'Compare another question' }));

    // Not cleared: Compare and Home treat a rated run as finished, and the
    // next question replaces it. Clearing it left Compare's settled answers
    // on screen with a link to a page that then had nothing to show.
    const stored = JSON.parse(window.sessionStorage.getItem(`sml.run.${COURSE_ID}`) ?? 'null');
    expect(stored?.ratedAt).toEqual(expect.any(String));
  });

  it('leaves any other save failure as a failure the student can retry', async () => {
    storeRun();
    addEvaluationMock.mockRejectedValue(new ApiError('The backend could not save the evaluation.', 503));
    renderPage();

    chooseAllCriteria();
    fireEvent.click(screen.getByRole('button', { name: 'Submit evaluation' }));

    await waitFor(() => {
      expect(addEvaluationMock).toHaveBeenCalledTimes(1);
    });
    expect(screen.getByRole('button', { name: 'Submit evaluation' })).toBeInTheDocument();
    const stored = JSON.parse(window.sessionStorage.getItem(`sml.run.${COURSE_ID}`) ?? 'null');
    expect(stored.ratedAt).toBeUndefined();
  });
});
