/** @vitest-environment jsdom */
import { cleanup, render, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { CourseProvider } from '../context/CourseContext';
import { generateBaseModel, generateRag } from '../lib/api';
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
  generateRag: vi.fn(),
}));

const generateBaseModelMock = vi.mocked(generateBaseModel);
const generateRagMock = vi.mocked(generateRag);

describe('ComparisonPage course-specific live requests', () => {
  beforeEach(() => {
    generateBaseModelMock.mockReset();
    generateRagMock.mockReset();
    generateBaseModelMock.mockResolvedValue({
      answer: 'Base answer',
      model: 'llama3.2:3b',
      responseType: 'base',
      courseId: 'css-430-summer-2026-ibce',
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

  it('sends the route courseId to Base and RAG live requests', async () => {
    const courseId = 'css-430-summer-2026-ibce';

    render(
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

    await waitFor(() => {
      expect(generateBaseModelMock).toHaveBeenCalled();
      expect(generateRagMock).toHaveBeenCalled();
    });

    expect(generateBaseModelMock.mock.calls[0]?.[0]).toBe(courseId);
    expect(generateRagMock.mock.calls[0]?.[0]).toBe(courseId);
  });
});
