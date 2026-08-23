/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { ComparisonRunProvider } from '../../context/ComparisonRunContext';
import { CourseProvider } from '../../context/CourseContext';
import { ApiError, generateBaseModel, generateFineTuned, generateFineTunedRag, generateRag } from '../../lib/api';
import { ComparePage } from './ComparePage';

/*
 * Two suggested questions for this course. Suggestions come from the course's
 * own approved examples now, so the text is arbitrary — what these tests care
 * about is that the exact chosen question is what gets submitted.
 */
const SUGGESTED = vi.hoisted(() => [
  'What should I do if I know I will miss class?',
  'Is there a required textbook?',
]);

vi.mock('../../lib/api', () => ({
  ApiError: class ApiError extends Error {
    status?: number;

    constructor(message: string, status?: number) {
      super(message);
      this.name = 'ApiError';
      this.status = status;
    }
  },
  generateBaseModel: vi.fn(),
  generateFineTuned: vi.fn(),
  generateFineTunedRag: vi.fn(),
  generateRag: vi.fn(),
  // Suggestions now come from the course's own approved examples. These stand
  // in for them so the chips carry the questions this file already asserts on.
  listCourseSeeds: vi.fn(async () => ({
    courseId: 'css-430-summer-2026-ibce',
    count: 2,
    seeds: [
      { id: 's1', question: SUGGESTED[0], answer: 'a', reviewStatus: 'approved' },
      { id: 's2', question: SUGGESTED[1], answer: 'a', reviewStatus: 'approved' },
    ],
  })),
}));

const generateBaseModelMock = vi.mocked(generateBaseModel);
const generateFineTunedMock = vi.mocked(generateFineTuned);
const generateFineTunedRagMock = vi.mocked(generateFineTunedRag);
const generateRagMock = vi.mocked(generateRag);

const FIRST_QUESTION = SUGGESTED[0];
const SECOND_QUESTION = SUGGESTED[1];
const COURSE_ID = 'css-430-summer-2026-ibce';

function createDeferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function mockSuccessfulResponses(overrides?: {
  baseAnswer?: string;
  ragAnswer?: string;
  fineTunedAnswer?: string;
  fineTunedRagAnswer?: string;
}) {
  generateBaseModelMock.mockResolvedValue({
    answer: overrides?.baseAnswer ?? 'Base answer',
    model: 'llama3.2:3b',
    responseType: 'base',
    courseId: COURSE_ID,
  });
  generateFineTunedMock.mockResolvedValue({
    answer: overrides?.fineTunedAnswer ?? 'Fine-tuned answer',
    model: 'meta-llama/Llama-3.2-3B-Instruct',
    responseType: 'fineTuned',
    courseId: COURSE_ID,
    adapterLoaded: true,
    generationSeconds: 0.8,
  });
  generateFineTunedRagMock.mockResolvedValue({
    courseId: COURSE_ID,
    answer: overrides?.fineTunedRagAnswer ?? 'Fine-tuned RAG answer',
    model: 'meta-llama/Llama-3.2-3B-Instruct',
    responseType: 'fineTunedRag',
    adapterLoaded: true,
    generationSeconds: 1.0,
    sources: [
      {
        chunkId: 'css430-late-1',
        sectionTitle: 'CSS 430 Late Policy',
        text: 'Late policy excerpt',
        score: 0.91,
      },
    ],
    retrievedChunks: [],
  });
  generateRagMock.mockResolvedValue({
    courseId: COURSE_ID,
    answer: overrides?.ragAnswer ?? 'RAG answer',
    model: 'llama3.2:3b',
    responseType: 'rag',
    sources: [
      {
        chunkId: 'css430-late-1',
        sectionTitle: 'CSS 430 Late Policy',
        text: 'Late policy excerpt',
        score: 0.88,
      },
    ],
    retrievedChunks: [],
  });
}

function renderComparisonPage(courseId = COURSE_ID) {
  return render(
    <MemoryRouter initialEntries={[`/course/${courseId}/compare`]}>
      <Routes>
        <Route
          path="/course/:courseId/compare"
          element={
            <ComparisonRunProvider>
              <CourseProvider courseId={courseId}>
                <ComparePage />
              </CourseProvider>
            </ComparisonRunProvider>
          }
        />
      </Routes>
    </MemoryRouter>,
  );
}

/** The question box. Free text is the primary way to ask. */
function questionInput() {
  return screen.getByLabelText('What would you like to ask about this course?');
}

/** The single submit control; its label changes to "Asking…" while running. */
function askButton() {
  return screen.getByRole('button', { name: /^(Ask|Asking…)$/ });
}

/** A suggestion chip carries the example question as its label. */
function exampleChip(question: string) {
  return screen.getByRole('button', { name: question });
}

/**
 * Suggestions come from the course's approved examples, so they arrive one
 * tick after render. Await this before clicking a chip.
 */
async function suggestionsReady() {
  await screen.findByRole('button', { name: SUGGESTED[0] });
}

function askCustomQuestion(question: string) {
  fireEvent.change(questionInput(), { target: { value: question } });
  fireEvent.click(askButton());
}

function expectNoInferenceRequests() {
  expect(generateBaseModelMock).not.toHaveBeenCalled();
  expect(generateRagMock).not.toHaveBeenCalled();
  expect(generateFineTunedMock).not.toHaveBeenCalled();
  expect(generateFineTunedRagMock).not.toHaveBeenCalled();
}

describe('ComparePage manual comparison runs', () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    generateBaseModelMock.mockReset();
    generateFineTunedMock.mockReset();
    generateFineTunedRagMock.mockReset();
    generateRagMock.mockReset();
    mockSuccessfulResponses();
  });

  afterEach(() => {
    cleanup();
  });

  function getActiveQuestionStatus() {
    return screen.getByRole('status', { name: 'Active question' });
  }

  it('does not call any model endpoint on page load', () => {
    renderComparisonPage();

    // Nothing has been asked, so no question is shown yet.
    expect(screen.queryByRole('status', { name: 'Active question' })).toBeNull();
    expectNoInferenceRequests();
  });

  it('does not call any model endpoint while a question is being typed', () => {
    renderComparisonPage();

    // Nothing has been asked, so no question is shown yet.
    expect(screen.queryByRole('status', { name: 'Active question' })).toBeNull();
    expectNoInferenceRequests();
  });

  it('submits exactly the example question when its chip is clicked', async () => {
    renderComparisonPage();
    await suggestionsReady();

    fireEvent.click(exampleChip(SECOND_QUESTION));

    await waitFor(() => {
      expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
      expect(generateRagMock).toHaveBeenCalledTimes(1);
      expect(generateFineTunedMock).toHaveBeenCalledTimes(1);
      expect(generateFineTunedRagMock).toHaveBeenCalledTimes(1);
    });

    expect(generateBaseModelMock).toHaveBeenCalledWith(COURSE_ID, SECOND_QUESTION);
    expect(generateRagMock).toHaveBeenCalledWith(COURSE_ID, SECOND_QUESTION);
    expect(generateFineTunedMock).toHaveBeenCalledWith(COURSE_ID, SECOND_QUESTION);
    expect(generateFineTunedRagMock).toHaveBeenCalledWith(COURSE_ID, SECOND_QUESTION);
    expect(getActiveQuestionStatus()).toHaveTextContent(SECOND_QUESTION);
  });

  it('submits the typed question when Ask is clicked', async () => {
    renderComparisonPage();

    const customQuestion = 'What is the passing grade for this course?';
    askCustomQuestion(customQuestion);

    await waitFor(() => {
      expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
    });

    expect(generateBaseModelMock).toHaveBeenCalledWith(COURSE_ID, customQuestion);
    expect(generateRagMock).toHaveBeenCalledWith(COURSE_ID, customQuestion);
    expect(generateFineTunedMock).toHaveBeenCalledWith(COURSE_ID, customQuestion);
    expect(generateFineTunedRagMock).toHaveBeenCalledWith(COURSE_ID, customQuestion);
    expect(getActiveQuestionStatus()).toHaveTextContent(customQuestion);
  });

  it('disables submit buttons and shows a loading label while a comparison is running', async () => {
    const deferredBase = createDeferred<{
      answer: string;
      model: string;
      responseType: 'base';
      courseId: string;
    }>();
    generateBaseModelMock.mockReturnValue(deferredBase.promise);
    generateRagMock.mockReturnValue(new Promise(() => {}));
    generateFineTunedMock.mockReturnValue(new Promise(() => {}));
    generateFineTunedRagMock.mockReturnValue(new Promise(() => {}));

    renderComparisonPage();
    await suggestionsReady();

    fireEvent.click(exampleChip(FIRST_QUESTION));

    await waitFor(() => {
      expect(askButton()).toBeDisabled();
      expect(exampleChip(FIRST_QUESTION)).toBeDisabled();
    });

    deferredBase.resolve({
      answer: 'Base answer',
      model: 'llama3.2:3b',
      responseType: 'base',
      courseId: COURSE_ID,
    });
  });

  it('blocks a second example submission while a comparison is running', async () => {
    generateBaseModelMock.mockReturnValue(new Promise(() => {}));
    generateRagMock.mockReturnValue(new Promise(() => {}));
    generateFineTunedMock.mockReturnValue(new Promise(() => {}));
    generateFineTunedRagMock.mockReturnValue(new Promise(() => {}));

    renderComparisonPage();
    await suggestionsReady();

    fireEvent.click(exampleChip(FIRST_QUESTION));

    await waitFor(() => {
      expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
      expect(askButton()).toBeDisabled();
    });

    const runButton = askButton();
    expect(runButton).toBeDisabled();
    fireEvent.click(runButton);

    expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
    expect(generateBaseModelMock).toHaveBeenCalledWith(COURSE_ID, FIRST_QUESTION);
    expect(getActiveQuestionStatus()).toHaveTextContent(FIRST_QUESTION);
  });

  it('blocks a typed submission while an example comparison is running', async () => {
    generateBaseModelMock.mockReturnValue(new Promise(() => {}));
    generateRagMock.mockReturnValue(new Promise(() => {}));
    generateFineTunedMock.mockReturnValue(new Promise(() => {}));
    generateFineTunedRagMock.mockReturnValue(new Promise(() => {}));

    renderComparisonPage();
    await suggestionsReady();

    fireEvent.click(exampleChip(FIRST_QUESTION));

    await waitFor(() => {
      expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
      expect(exampleChip(FIRST_QUESTION)).toBeDisabled();
    });

    fireEvent.change(questionInput(), {
      target: { value: 'What is the passing grade for this course?' },
    });
    fireEvent.click(askButton());

    expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
    expect(generateBaseModelMock).toHaveBeenCalledWith(COURSE_ID, FIRST_QUESTION);
  });

  it('blocks an example submission while a typed comparison is running', async () => {
    generateBaseModelMock.mockReturnValue(new Promise(() => {}));
    generateRagMock.mockReturnValue(new Promise(() => {}));
    generateFineTunedMock.mockReturnValue(new Promise(() => {}));
    generateFineTunedRagMock.mockReturnValue(new Promise(() => {}));

    renderComparisonPage();

    const customQuestion = 'What is the passing grade for this course?';
    askCustomQuestion(customQuestion);

    await waitFor(() => {
      expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
      expect(askButton()).toBeDisabled();
    });

    fireEvent.click(exampleChip(SECOND_QUESTION));

    expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
    expect(generateBaseModelMock).toHaveBeenCalledWith(COURSE_ID, customQuestion);
    expect(getActiveQuestionStatus()).toHaveTextContent(customQuestion);
  });

  it('re-enables submission after the active comparison settles', async () => {
    const deferredBase = createDeferred<{
      answer: string;
      model: string;
      responseType: 'base';
      courseId: string;
    }>();
    const deferredRag = createDeferred<{
      courseId: string;
      answer: string;
      model: string;
      responseType: 'rag';
      sources: never[];
      retrievedChunks: never[];
    }>();
    const deferredFineTuned = createDeferred<{
      answer: string;
      model: string;
      responseType: 'fineTuned';
      courseId: string;
      adapterLoaded: boolean;
      generationSeconds: number;
    }>();
    const deferredFineTunedRag = createDeferred<{
      courseId: string;
      answer: string;
      model: string;
      responseType: 'fineTunedRag';
      adapterLoaded: boolean;
      generationSeconds: number;
      sources: never[];
      retrievedChunks: never[];
    }>();

    generateBaseModelMock.mockReturnValue(deferredBase.promise);
    generateRagMock.mockReturnValue(deferredRag.promise);
    generateFineTunedMock.mockReturnValue(deferredFineTuned.promise);
    generateFineTunedRagMock.mockReturnValue(deferredFineTunedRag.promise);

    renderComparisonPage();
    await suggestionsReady();

    fireEvent.click(exampleChip(FIRST_QUESTION));

    await waitFor(() => {
      expect(askButton()).toBeDisabled();
      expect(exampleChip(FIRST_QUESTION)).toBeDisabled();
    });

    deferredBase.resolve({
      answer: 'Base answer',
      model: 'llama3.2:3b',
      responseType: 'base',
      courseId: COURSE_ID,
    });
    deferredFineTuned.resolve({
      answer: 'Fine-tuned answer',
      model: 'meta-llama/Llama-3.2-3B-Instruct',
      responseType: 'fineTuned',
      courseId: COURSE_ID,
      adapterLoaded: true,
      generationSeconds: 0.8,
    });
    deferredFineTunedRag.resolve({
      courseId: COURSE_ID,
      answer: 'Fine-tuned RAG answer',
      model: 'meta-llama/Llama-3.2-3B-Instruct',
      responseType: 'fineTunedRag',
      adapterLoaded: true,
      generationSeconds: 1.0,
      sources: [],
      retrievedChunks: [],
    });

    // Buttons stay disabled until RAG (last local Ollama path) settles too.
    await waitFor(() => {
      expect(generateRagMock).toHaveBeenCalledTimes(1);
    });
    expect(askButton()).toBeDisabled();

    deferredRag.resolve({
      courseId: COURSE_ID,
      answer: 'RAG answer',
      model: 'llama3.2:3b',
      responseType: 'rag',
      sources: [],
      retrievedChunks: [],
    });

    await waitFor(() => {
      expect(exampleChip(FIRST_QUESTION)).not.toBeDisabled();
    });
  });

  it('starts Base before RAG and does not call RAG until Base settles', async () => {
    const callOrder: string[] = [];
    const deferredBase = createDeferred<{
      answer: string;
      model: string;
      responseType: 'base';
      courseId: string;
    }>();

    generateBaseModelMock.mockImplementation(() => {
      callOrder.push('base-start');
      return deferredBase.promise.then((value) => {
        callOrder.push('base-settle');
        return value;
      });
    });
    generateRagMock.mockImplementation(async () => {
      callOrder.push('rag-start');
      return {
        courseId: COURSE_ID,
        answer: 'RAG answer',
        model: 'llama3.2:3b',
        responseType: 'rag' as const,
        sources: [],
        retrievedChunks: [],
      };
    });

    renderComparisonPage();
    await suggestionsReady();
    fireEvent.click(exampleChip(FIRST_QUESTION));

    await waitFor(() => {
      expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
    });
    expect(generateRagMock).not.toHaveBeenCalled();
    expect(callOrder).toEqual(['base-start']);

    deferredBase.resolve({
      answer: 'Base answer',
      model: 'llama3.2:3b',
      responseType: 'base',
      courseId: COURSE_ID,
    });

    await waitFor(() => {
      expect(generateRagMock).toHaveBeenCalledTimes(1);
    });
    expect(callOrder).toEqual(['base-start', 'base-settle', 'rag-start']);
  });

  it('still runs RAG when Base fails', async () => {
    generateBaseModelMock.mockRejectedValue(new ApiError('Base unavailable', 503));
    generateRagMock.mockResolvedValue({
      courseId: COURSE_ID,
      answer: 'RAG recovered answer',
      model: 'llama3.2:3b',
      responseType: 'rag',
      sources: [],
      retrievedChunks: [],
    });

    renderComparisonPage();
    await suggestionsReady();
    fireEvent.click(exampleChip(FIRST_QUESTION));

    await waitFor(() => {
      expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
      expect(generateRagMock).toHaveBeenCalledTimes(1);
    });

    await waitFor(() => {
      // The student sees a friendly message, never the backend's wording.
      expect(
        screen.getByText('This response is temporarily unavailable. Try again in a moment.'),
      ).toBeInTheDocument();
      expect(screen.queryByText('Base unavailable')).not.toBeInTheDocument();
      expect(screen.getByText('RAG recovered answer')).toBeInTheDocument();
    });
  });

  it('keeps the Base result when RAG fails', async () => {
    generateBaseModelMock.mockResolvedValue({
      answer: 'Base survived answer',
      model: 'llama3.2:3b',
      responseType: 'base',
      courseId: COURSE_ID,
    });
    generateRagMock.mockRejectedValue(new ApiError('RAG unavailable', 503));

    renderComparisonPage();
    await suggestionsReady();
    fireEvent.click(exampleChip(FIRST_QUESTION));

    await waitFor(() => {
      expect(screen.getByText('Base survived answer')).toBeInTheDocument();
      expect(
        screen.getByText('This response is temporarily unavailable. Try again in a moment.'),
      ).toBeInTheDocument();
      expect(screen.queryByText('RAG unavailable')).not.toBeInTheDocument();
    });
  });

  it('keeps Fine-Tuned calls functional alongside sequential Base then RAG', async () => {
    const deferredBase = createDeferred<{
      answer: string;
      model: string;
      responseType: 'base';
      courseId: string;
    }>();

    generateBaseModelMock.mockReturnValue(deferredBase.promise);
    generateRagMock.mockResolvedValue({
      courseId: COURSE_ID,
      answer: 'RAG answer',
      model: 'llama3.2:3b',
      responseType: 'rag',
      sources: [],
      retrievedChunks: [],
    });

    renderComparisonPage();
    await suggestionsReady();
    fireEvent.click(exampleChip(FIRST_QUESTION));

    await waitFor(() => {
      expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
      expect(generateFineTunedMock).toHaveBeenCalledTimes(1);
      expect(generateFineTunedRagMock).toHaveBeenCalledTimes(1);
    });
    // Remote fine-tuned paths start without waiting for Base; RAG still waits.
    expect(generateRagMock).not.toHaveBeenCalled();

    deferredBase.resolve({
      answer: 'Base answer',
      model: 'llama3.2:3b',
      responseType: 'base',
      courseId: COURSE_ID,
    });

    await waitFor(() => {
      expect(generateRagMock).toHaveBeenCalledTimes(1);
      expect(screen.getByText('Fine-tuned answer')).toBeInTheDocument();
      expect(screen.getByText('Fine-tuned RAG answer')).toBeInTheDocument();
    });
  });

  it('does not let a prior comparison overwrite a newer run once the lock releases', async () => {
    mockSuccessfulResponses({
      baseAnswer: 'First base answer',
      ragAnswer: 'First rag answer',
      fineTunedAnswer: 'First fine-tuned answer',
      fineTunedRagAnswer: 'First fine-tuned RAG answer',
    });

    renderComparisonPage();
    await suggestionsReady();
    fireEvent.click(exampleChip(FIRST_QUESTION));

    await waitFor(() => {
      expect(screen.getByText('First base answer')).toBeInTheDocument();
      expect(exampleChip(FIRST_QUESTION)).not.toBeDisabled();
    });

    mockSuccessfulResponses({
      baseAnswer: 'Second base answer',
      ragAnswer: 'Second rag answer',
      fineTunedAnswer: 'Second fine-tuned answer',
      fineTunedRagAnswer: 'Second fine-tuned RAG answer',
    });

    fireEvent.click(exampleChip(SECOND_QUESTION));

    await waitFor(() => {
      expect(screen.getByText('Second base answer')).toBeInTheDocument();
      expect(screen.getByText('Second rag answer')).toBeInTheDocument();
    });
    expect(screen.queryByText('First base answer')).not.toBeInTheDocument();
    expect(screen.queryByText('First rag answer')).not.toBeInTheDocument();
    expect(generateBaseModelMock).toHaveBeenLastCalledWith(COURSE_ID, SECOND_QUESTION);
    expect(generateRagMock).toHaveBeenLastCalledWith(COURSE_ID, SECOND_QUESTION);
  });

  it('hands the settled run to Evaluate, including the failed approach', async () => {
    generateBaseModelMock.mockRejectedValue(new ApiError('Base down', 503));

    renderComparisonPage();
    await suggestionsReady();
    fireEvent.click(exampleChip(FIRST_QUESTION));

    await waitFor(() => {
      expect(
        screen.getByRole('link', { name: /Evaluate these responses/ }),
      ).toBeInTheDocument();
    });

    const stored = window.sessionStorage.getItem(`sml.run.${COURSE_ID}`);
    expect(stored).not.toBeNull();

    const run = JSON.parse(String(stored));
    expect(run.courseId).toBe(COURSE_ID);
    expect(run.question).toBe(FIRST_QUESTION);
    // Course-derived suggestions have no predefined-comparison id, so the run
    // is recorded against its question text instead.
    expect(run.matchedComparisonId).toBeNull();
    expect(run.responses.rag.text).toBe('RAG answer');
    // A failure is carried across as a student-facing message, not raw text.
    expect(run.responses.base.error).toContain('temporarily unavailable');
    expect(run.responses.base.error).not.toContain('Base down');
  });

  it('does not offer evaluation until every approach has settled', async () => {
    const deferredRag = createDeferred<{
      courseId: string;
      answer: string;
      model: string;
      responseType: 'rag';
      sources: never[];
      retrievedChunks: never[];
    }>();
    generateRagMock.mockReturnValue(deferredRag.promise);

    renderComparisonPage();
    await suggestionsReady();
    fireEvent.click(exampleChip(FIRST_QUESTION));

    await waitFor(() => {
      expect(generateRagMock).toHaveBeenCalledTimes(1);
    });
    expect(screen.queryByRole('link', { name: /Evaluate these responses/ })).toBeNull();
    expect(screen.getByText('Waiting for all four responses…')).toBeInTheDocument();

    deferredRag.resolve({
      courseId: COURSE_ID,
      answer: 'RAG answer',
      model: 'llama3.2:3b',
      responseType: 'rag',
      sources: [],
      retrievedChunks: [],
    });

    await waitFor(() => {
      expect(
        screen.getByRole('link', { name: /Evaluate these responses/ }),
      ).toBeInTheDocument();
    });
  });
});
