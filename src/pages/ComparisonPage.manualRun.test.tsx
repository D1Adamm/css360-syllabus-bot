/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import comparisonData from '../data/comparisonData.json';
import { CourseProvider } from '../context/CourseContext';
import { generateBaseModel, generateFineTuned, generateFineTunedRag, generateRag } from '../lib/api';
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

  it('ignores a stale comparison response after a newer comparison is submitted', async () => {
    const firstBase = createDeferred<{
      answer: string;
      model: string;
      responseType: 'base';
      courseId: string;
    }>();
    const secondBase = createDeferred<{
      answer: string;
      model: string;
      responseType: 'base';
      courseId: string;
    }>();

    generateBaseModelMock
      .mockReturnValueOnce(firstBase.promise)
      .mockReturnValueOnce(secondBase.promise);
    generateRagMock
      .mockResolvedValueOnce({
        courseId: COURSE_ID,
        answer: 'Old RAG answer',
        model: 'llama3.2:3b',
        responseType: 'rag',
        sources: [],
        retrievedChunks: [],
      })
      .mockResolvedValueOnce({
        courseId: COURSE_ID,
        answer: 'New RAG answer',
        model: 'llama3.2:3b',
        responseType: 'rag',
        sources: [],
        retrievedChunks: [],
      });
    generateFineTunedMock
      .mockResolvedValueOnce({
        answer: 'Old fine-tuned answer',
        model: 'meta-llama/Llama-3.2-3B-Instruct',
        responseType: 'fineTuned',
        courseId: COURSE_ID,
        adapterLoaded: true,
        generationSeconds: 0.8,
      })
      .mockResolvedValueOnce({
        answer: 'New fine-tuned answer',
        model: 'meta-llama/Llama-3.2-3B-Instruct',
        responseType: 'fineTuned',
        courseId: COURSE_ID,
        adapterLoaded: true,
        generationSeconds: 0.8,
      });
    generateFineTunedRagMock
      .mockResolvedValueOnce({
        courseId: COURSE_ID,
        answer: 'Old fine-tuned RAG answer',
        model: 'meta-llama/Llama-3.2-3B-Instruct',
        responseType: 'fineTunedRag',
        adapterLoaded: true,
        generationSeconds: 1.0,
        sources: [],
        retrievedChunks: [],
      })
      .mockResolvedValueOnce({
        courseId: COURSE_ID,
        answer: 'New fine-tuned RAG answer',
        model: 'meta-llama/Llama-3.2-3B-Instruct',
        responseType: 'fineTunedRag',
        adapterLoaded: true,
        generationSeconds: 1.0,
        sources: [],
        retrievedChunks: [],
      });

    renderComparisonPage();

    fireEvent.click(screen.getByRole('button', { name: 'Run comparison' }));

    await waitFor(() => {
      expect(generateBaseModelMock).toHaveBeenCalledTimes(1);
      expect(getActiveQuestionStatus()).toHaveTextContent(
        `Active question: ${FIRST_QUESTION}`,
      );
    });

    fireEvent.change(screen.getByLabelText('Predefined syllabus questions'), {
      target: { value: records[1]?.id },
    });

    const runButton = screen.getByRole('button', { name: 'Run comparison' });
    expect(runButton).not.toBeDisabled();
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(generateBaseModelMock).toHaveBeenCalledTimes(2);
      expect(getActiveQuestionStatus()).toHaveTextContent(
        `Active question: ${SECOND_QUESTION}`,
      );
    });

    secondBase.resolve({
      answer: 'New base answer',
      model: 'llama3.2:3b',
      responseType: 'base',
      courseId: COURSE_ID,
    });

    await waitFor(() => {
      expect(screen.getByText('New base answer')).toBeInTheDocument();
    });

    firstBase.resolve({
      answer: 'Old base answer',
      model: 'llama3.2:3b',
      responseType: 'base',
      courseId: COURSE_ID,
    });

    await waitFor(() => {
      expect(generateBaseModelMock).toHaveBeenCalledTimes(2);
    });

    expect(screen.getByText('New base answer')).toBeInTheDocument();
    expect(screen.queryByText('Old base answer')).not.toBeInTheDocument();
    expect(getActiveQuestionStatus()).toHaveTextContent(
      `Active question: ${SECOND_QUESTION}`,
    );
  });
});
