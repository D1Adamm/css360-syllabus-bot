/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { ComparisonRunProvider } from '../../context/ComparisonRunContext';
import { CourseProvider } from '../../context/CourseContext';
import { generateBaseModel, generateFineTuned, generateFineTunedRag, generateRag } from '../../lib/api';
import { ComparePage } from './ComparePage';

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
  listCourseSeeds: vi.fn(async () => ({
    courseId: 'css-430-summer-2026-ibce',
    count: 0,
    firebasePath: '',
    seeds: [],
  })),
}));

const generateBaseModelMock = vi.mocked(generateBaseModel);
const generateFineTunedMock = vi.mocked(generateFineTuned);
const generateFineTunedRagMock = vi.mocked(generateFineTunedRag);
const generateRagMock = vi.mocked(generateRag);

describe('ComparePage course-specific live requests', () => {
  beforeEach(() => {
    generateBaseModelMock.mockReset();
    generateFineTunedMock.mockReset();
    generateFineTunedRagMock.mockReset();
    generateRagMock.mockReset();
    generateBaseModelMock.mockResolvedValue({
      answer: 'Base answer',
      model: 'llama3.2:3b',
      responseType: 'base',
      courseId: 'css-430-summer-2026-ibce',
    });
    generateFineTunedMock.mockResolvedValue({
      answer: 'Fine-tuned answer',
      model: 'meta-llama/Llama-3.2-3B-Instruct',
      responseType: 'fineTuned',
      courseId: 'css-430-summer-2026-ibce',
      adapterLoaded: true,
      generationSeconds: 0.8,
    });
    generateFineTunedRagMock.mockResolvedValue({
      courseId: 'css-430-summer-2026-ibce',
      answer: 'Fine-tuned RAG answer',
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
      courseId: 'css-430-summer-2026-ibce',
      answer: 'RAG answer',
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
  });

  afterEach(() => {
    cleanup();
  });

  it('sends the route courseId to all four live model requests', async () => {
    const courseId = 'css-430-summer-2026-ibce';

    render(
      <MemoryRouter initialEntries={[`/course/${courseId}/compare`]}>
        <Routes>
          <Route
            path="/course/:courseId/compare"
            element={
              <ComparisonRunProvider>
                <ComparisonRunProvider>
                  <CourseProvider courseId={courseId}>
                    <ComparePage />
                  </CourseProvider>
                </ComparisonRunProvider>
              </ComparisonRunProvider>
            }
          />
        </Routes>
      </MemoryRouter>,
    );

    expect(generateBaseModelMock).not.toHaveBeenCalled();

    fireEvent.change(
      screen.getByLabelText('What would you like to ask about this course?'),
      { target: { value: 'What is the late policy?' } },
    );
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }));

    await waitFor(() => {
      expect(generateBaseModelMock).toHaveBeenCalled();
      expect(generateRagMock).toHaveBeenCalled();
      expect(generateFineTunedMock).toHaveBeenCalled();
      expect(generateFineTunedRagMock).toHaveBeenCalled();
    });

    expect(generateBaseModelMock.mock.calls[0]?.[0]).toBe(courseId);
    expect(generateRagMock.mock.calls[0]?.[0]).toBe(courseId);
    expect(generateFineTunedMock.mock.calls[0]?.[0]).toBe(courseId);
    expect(generateFineTunedRagMock.mock.calls[0]?.[0]).toBe(courseId);
  });
});
