/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import comparisonData from '../data/comparisonData.json';
import { CourseProvider } from '../context/CourseContext';
import { ApiError, generateBaseModel, generateFineTuned, generateFineTunedRag, generateRag } from '../lib/api';
import type { ComparisonRecord } from '../types';
import { ComparisonPage } from './ComparisonPage';

vi.mock('../lib/api', () => ({
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
}));

const generateBaseModelMock = vi.mocked(generateBaseModel);
const generateFineTunedMock = vi.mocked(generateFineTuned);
const generateFineTunedRagMock = vi.mocked(generateFineTunedRag);
const generateRagMock = vi.mocked(generateRag);

const records = comparisonData as ComparisonRecord[];
const FIRST_QUESTION = records[0]?.question ?? '';
const SECOND_QUESTION = records[1]?.question ?? '';
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
            <CourseProvider courseId={courseId}>
              <ComparisonPage />
            </CourseProvider>
          }
        />
      </Routes>
    </MemoryRouter>,
  );
}

function expectNoInferenceRequests() {
  expect(generateBaseModelMock).not.toHaveBeenCalled();
  expect(generateRagMock).not.toHaveBeenCalled();
  expect(generateFineTunedMock).not.toHaveBeenCalled();
  expect(generateFineTunedRagMock).not.toHaveBeenCalled();
}

describe('ComparisonPage manual comparison runs', () => {
  beforeEach(() => {
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

    expect(getActiveQuestionStatus()).toHaveTextContent('Active question: None yet');
    expectNoInferenceRequests();
  });

  it('does not call any model endpoint when the predefined question dropdown changes', () => {
    renderComparisonPage();

    fireEvent.change(screen.getByLabelText('Predefined syllabus questions'), {
      target: { value: records[1]?.id },
    });

    expect(screen.getByText('Course Basics')).toBeInTheDocument();
    expect(screen.getByText('Textbook')).toBeInTheDocument();
    expect(getActiveQuestionStatus()).toHaveTextContent('Active question: None yet');
    expectNoInferenceRequests();
  });

  it('submits exactly the selected predefined question when Run comparison is clicked', async () => {
    renderComparisonPage();

    fireEvent.change(screen.getByLabelText('Predefined syllabus questions'), {
      target: { value: records[1]?.id },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Run comparison' }));

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
    expect(getActiveQuestionStatus()).toHaveTextContent(
      `Active question: ${SECOND_QUESTION}`,
    );
  });

  it('submits the typed custom question when Ask question is clicked', async () => {
    renderComparisonPage();

    const customQuestion = 'What is the passing grade for this course?';
    fireEvent.change(screen.getByLabelText('Enter a question to send to the live models'), {
      target: { value: customQuestion },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Ask question' }));

    await waitFor(() => {
      expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
    });

    expect(generateBaseModelMock).toHaveBeenCalledWith(COURSE_ID, customQuestion);
    expect(generateRagMock).toHaveBeenCalledWith(COURSE_ID, customQuestion);
    expect(generateFineTunedMock).toHaveBeenCalledWith(COURSE_ID, customQuestion);
    expect(generateFineTunedRagMock).toHaveBeenCalledWith(COURSE_ID, customQuestion);
    expect(getActiveQuestionStatus()).toHaveTextContent(
      `Active question: ${customQuestion}`,
    );
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

    fireEvent.click(screen.getByRole('button', { name: 'Run comparison' }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Running comparison…' })).toBeDisabled();
      expect(screen.getByRole('button', { name: 'Generating...' })).toBeDisabled();
    });

    deferredBase.resolve({
      answer: 'Base answer',
      model: 'llama3.2:3b',
      responseType: 'base',
      courseId: COURSE_ID,
    });
  });

  it('blocks a second predefined submission while a comparison is running', async () => {
    generateBaseModelMock.mockReturnValue(new Promise(() => {}));
    generateRagMock.mockReturnValue(new Promise(() => {}));
    generateFineTunedMock.mockReturnValue(new Promise(() => {}));
    generateFineTunedRagMock.mockReturnValue(new Promise(() => {}));

    renderComparisonPage();

    fireEvent.click(screen.getByRole('button', { name: 'Run comparison' }));

    await waitFor(() => {
      expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
      expect(screen.getByRole('button', { name: 'Running comparison…' })).toBeDisabled();
    });

    fireEvent.change(screen.getByLabelText('Predefined syllabus questions'), {
      target: { value: records[1]?.id },
    });

    const runButton = screen.getByRole('button', { name: 'Running comparison…' });
    expect(runButton).toBeDisabled();
    fireEvent.click(runButton);

    expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
    expect(generateBaseModelMock).toHaveBeenCalledWith(COURSE_ID, FIRST_QUESTION);
    expect(getActiveQuestionStatus()).toHaveTextContent(
      `Active question: ${FIRST_QUESTION}`,
    );
  });

  it('blocks a custom submission while a predefined comparison is running', async () => {
    generateBaseModelMock.mockReturnValue(new Promise(() => {}));
    generateRagMock.mockReturnValue(new Promise(() => {}));
    generateFineTunedMock.mockReturnValue(new Promise(() => {}));
    generateFineTunedRagMock.mockReturnValue(new Promise(() => {}));

    renderComparisonPage();

    fireEvent.click(screen.getByRole('button', { name: 'Run comparison' }));

    await waitFor(() => {
      expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
      expect(screen.getByRole('button', { name: 'Generating...' })).toBeDisabled();
    });

    fireEvent.change(screen.getByLabelText('Enter a question to send to the live models'), {
      target: { value: 'What is the passing grade for this course?' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Generating...' }));

    expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
    expect(generateBaseModelMock).toHaveBeenCalledWith(COURSE_ID, FIRST_QUESTION);
  });

  it('blocks a predefined submission while a custom comparison is running', async () => {
    generateBaseModelMock.mockReturnValue(new Promise(() => {}));
    generateRagMock.mockReturnValue(new Promise(() => {}));
    generateFineTunedMock.mockReturnValue(new Promise(() => {}));
    generateFineTunedRagMock.mockReturnValue(new Promise(() => {}));

    renderComparisonPage();

    const customQuestion = 'What is the passing grade for this course?';
    fireEvent.change(screen.getByLabelText('Enter a question to send to the live models'), {
      target: { value: customQuestion },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Ask question' }));

    await waitFor(() => {
      expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
      expect(screen.getByRole('button', { name: 'Running comparison…' })).toBeDisabled();
    });

    fireEvent.change(screen.getByLabelText('Predefined syllabus questions'), {
      target: { value: records[1]?.id },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Running comparison…' }));

    expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
    expect(generateBaseModelMock).toHaveBeenCalledWith(COURSE_ID, customQuestion);
    expect(getActiveQuestionStatus()).toHaveTextContent(
      `Active question: ${customQuestion}`,
    );
  });

  it('re-enables both submission buttons after the active comparison settles', async () => {
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

    fireEvent.click(screen.getByRole('button', { name: 'Run comparison' }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Running comparison…' })).toBeDisabled();
      expect(screen.getByRole('button', { name: 'Generating...' })).toBeDisabled();
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
    expect(screen.getByRole('button', { name: 'Running comparison…' })).toBeDisabled();

    deferredRag.resolve({
      courseId: COURSE_ID,
      answer: 'RAG answer',
      model: 'llama3.2:3b',
      responseType: 'rag',
      sources: [],
      retrievedChunks: [],
    });

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Run comparison' })).not.toBeDisabled();
      expect(screen.getByRole('button', { name: 'Ask question' })).not.toBeDisabled();
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
    fireEvent.click(screen.getByRole('button', { name: 'Run comparison' }));

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
    fireEvent.click(screen.getByRole('button', { name: 'Run comparison' }));

    await waitFor(() => {
      expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
      expect(generateRagMock).toHaveBeenCalledTimes(1);
    });

    await waitFor(() => {
      expect(screen.getByText('Base unavailable')).toBeInTheDocument();
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
    fireEvent.click(screen.getByRole('button', { name: 'Run comparison' }));

    await waitFor(() => {
      expect(screen.getByText('Base survived answer')).toBeInTheDocument();
      expect(screen.getByText('RAG unavailable')).toBeInTheDocument();
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
    fireEvent.click(screen.getByRole('button', { name: 'Run comparison' }));

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
    fireEvent.click(screen.getByRole('button', { name: 'Run comparison' }));

    await waitFor(() => {
      expect(screen.getByText('First base answer')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Run comparison' })).not.toBeDisabled();
    });

    mockSuccessfulResponses({
      baseAnswer: 'Second base answer',
      ragAnswer: 'Second rag answer',
      fineTunedAnswer: 'Second fine-tuned answer',
      fineTunedRagAnswer: 'Second fine-tuned RAG answer',
    });

    fireEvent.change(screen.getByLabelText('Predefined syllabus questions'), {
      target: { value: records[1]?.id },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Run comparison' }));

    await waitFor(() => {
      expect(screen.getByText('Second base answer')).toBeInTheDocument();
      expect(screen.getByText('Second rag answer')).toBeInTheDocument();
    });
    expect(screen.queryByText('First base answer')).not.toBeInTheDocument();
    expect(screen.queryByText('First rag answer')).not.toBeInTheDocument();
    expect(generateBaseModelMock).toHaveBeenLastCalledWith(COURSE_ID, SECOND_QUESTION);
    expect(generateRagMock).toHaveBeenLastCalledWith(COURSE_ID, SECOND_QUESTION);
  });
});
