/** @vitest-environment jsdom */
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

const {
  subscribeToSeedExamplesMock,
  createSeedExampleMock,
  deleteSeedExampleMock,
  deleteAllSeedExamplesMock,
  deleteAllUserSeedExamplesMock,
  updateSeedExampleMock,
  subscribeToEvaluationsMock,
  createEvaluationMock,
  deleteEvaluationMock,
  deleteAllEvaluationsMock,
} = vi.hoisted(() => ({
  subscribeToSeedExamplesMock: vi.fn(),
  createSeedExampleMock: vi.fn(),
  deleteSeedExampleMock: vi.fn(),
  deleteAllSeedExamplesMock: vi.fn(),
  deleteAllUserSeedExamplesMock: vi.fn(),
  updateSeedExampleMock: vi.fn(),
  subscribeToEvaluationsMock: vi.fn(),
  createEvaluationMock: vi.fn(),
  deleteEvaluationMock: vi.fn(),
  deleteAllEvaluationsMock: vi.fn(),
}));

vi.mock('../lib/firebase', () => ({
  app: {},
  database: { name: 'mock-db' },
}));

vi.mock('firebase/database', () => ({
  ref: vi.fn(),
  onValue: vi.fn(() => () => undefined),
  push: vi.fn(() => ({ key: 'generated-id' })),
  set: vi.fn(async () => undefined),
  remove: vi.fn(async () => undefined),
  get: vi.fn(async () => ({ exists: () => false, val: () => null })),
  update: vi.fn(async () => undefined),
}));

vi.mock('../lib/seedExamplesDb', async () => {
  const actual = await vi.importActual<typeof import('../lib/seedExamplesDb')>(
    '../lib/seedExamplesDb',
  );
  return {
    ...actual,
    subscribeToSeedExamples: subscribeToSeedExamplesMock,
    createSeedExample: createSeedExampleMock,
    deleteSeedExample: deleteSeedExampleMock,
    deleteAllSeedExamples: deleteAllSeedExamplesMock,
    deleteAllUserSeedExamples: deleteAllUserSeedExamplesMock,
    updateSeedExample: updateSeedExampleMock,
  };
});

vi.mock('../lib/evaluationsDb', async () => {
  const actual = await vi.importActual<typeof import('../lib/evaluationsDb')>(
    '../lib/evaluationsDb',
  );
  return {
    ...actual,
    subscribeToEvaluations: subscribeToEvaluationsMock,
    createEvaluation: createEvaluationMock,
    deleteEvaluation: deleteEvaluationMock,
    deleteAllEvaluations: deleteAllEvaluationsMock,
  };
});

vi.mock('../lib/api', () => ({
  ApiError: class ApiError extends Error {},
  generateBaseModel: vi.fn(),
  generateFineTuned: vi.fn(),
  generateRag: vi.fn(),
}));

import { CourseProvider } from '../context/CourseContext';
import { getCourseEvaluationsPath, getCourseSeedExamplesPath } from '../lib/coursePaths';
import { SeedBuilderPage } from '../pages/SeedBuilderPage';
import { SeedDatasetPage } from '../pages/SeedDatasetPage';
import { EvaluationPage } from '../pages/EvaluationPage';
import { ResultsPage } from '../pages/ResultsPage';

function mockSeedSubscription(courseId: string) {
  subscribeToSeedExamplesMock.mockImplementation(
    (
      receivedCourseId: string,
      onData: (seeds: unknown[]) => void,
      _onError: (message: string) => void,
    ) => {
      expect(receivedCourseId).toBe(courseId);
      onData([]);
      return () => undefined;
    },
  );
}

function mockEvaluationSubscription(courseId: string) {
  subscribeToEvaluationsMock.mockImplementation(
    (
      receivedCourseId: string,
      onData: (evaluations: unknown[]) => void,
      _onError: (message: string) => void,
    ) => {
      expect(receivedCourseId).toBe(courseId);
      onData([]);
      return () => undefined;
    },
  );
}

function renderWithCourse(courseId: string, page: React.ReactNode, pathSuffix: string) {
  return render(
    <MemoryRouter initialEntries={[`/course/${courseId}/${pathSuffix}`]}>
      <Routes>
        <Route
          path={`/course/${courseId}/${pathSuffix}`}
          element={<CourseProvider courseId={courseId}>{page}</CourseProvider>}
        />
      </Routes>
    </MemoryRouter>,
  );
}

describe('course-scoped seed and evaluation data', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    createSeedExampleMock.mockResolvedValue(undefined);
    createEvaluationMock.mockResolvedValue({
      id: 'eval-1',
      comparisonId: 'comparison-001',
      mostAccurate: 'rag',
      mostHelpful: 'rag',
      mostConcise: 'rag',
      bestGrounded: 'rag',
      preferredModel: 'rag',
      hallucinationFlags: [],
      createdAt: '2026-01-01T00:00:00.000Z',
    });
  });

  afterEach(() => {
    cleanup();
  });

  it('Seed Data page subscribes to courses/{courseId}/seedExamples', async () => {
    const courseId = 'css360-default';
    mockSeedSubscription(courseId);

    renderWithCourse(courseId, <SeedBuilderPage />, 'seeds');

    await waitFor(() => {
      expect(subscribeToSeedExamplesMock).toHaveBeenCalledWith(
        courseId,
        expect.any(Function),
        expect.any(Function),
      );
    });

    expect(getCourseSeedExamplesPath(courseId)).toBe(
      `courses/${courseId}/seedExamples`,
    );
    expect(
      document.body.textContent,
    ).toMatch(/No manually created examples yet/);
  });

  it('Dataset page uses the same course-specific seed path', async () => {
    const courseId = 'css360-default';
    mockSeedSubscription(courseId);

    renderWithCourse(courseId, <SeedDatasetPage />, 'dataset');

    await waitFor(() => {
      expect(subscribeToSeedExamplesMock).toHaveBeenCalledWith(
        courseId,
        expect.any(Function),
        expect.any(Function),
      );
    });
    expect(getCourseSeedExamplesPath(courseId)).toBe(
      'courses/css360-default/seedExamples',
    );
    expect(
      await screen.findByText('No seed examples have been created for this course yet.'),
    ).toBeInTheDocument();
  });

  it('Results reads from courses/{courseId}/evaluations', async () => {
    const courseId = 'css360-default';
    mockEvaluationSubscription(courseId);

    renderWithCourse(courseId, <ResultsPage />, 'results');

    await waitFor(() => {
      expect(subscribeToEvaluationsMock).toHaveBeenCalledWith(
        courseId,
        expect.any(Function),
        expect.any(Function),
      );
    });
    expect(getCourseEvaluationsPath(courseId)).toBe(
      'courses/css360-default/evaluations',
    );
    expect(document.body.textContent).toMatch(/No evaluations for this course yet/);
  });

  it('Evaluate page subscribes to course-specific evaluations for saves', async () => {
    const courseId = 'css360-default';
    mockEvaluationSubscription(courseId);

    renderWithCourse(courseId, <EvaluationPage />, 'evaluate');

    await waitFor(() => {
      expect(subscribeToEvaluationsMock).toHaveBeenCalledWith(
        courseId,
        expect.any(Function),
        expect.any(Function),
      );
    });
  });

  it('does not share seed examples across different course ids', async () => {
    mockSeedSubscription('course-alpha');
    const first = renderWithCourse('course-alpha', <SeedBuilderPage />, 'seeds');
    await waitFor(() => {
      expect(subscribeToSeedExamplesMock).toHaveBeenCalledWith(
        'course-alpha',
        expect.any(Function),
        expect.any(Function),
      );
    });
    first.unmount();

    mockSeedSubscription('course-beta');
    renderWithCourse('course-beta', <SeedBuilderPage />, 'seeds');
    await waitFor(() => {
      expect(subscribeToSeedExamplesMock).toHaveBeenCalledWith(
        'course-beta',
        expect.any(Function),
        expect.any(Function),
      );
    });

    const seedCourseIds = subscribeToSeedExamplesMock.mock.calls.map((call) => call[0]);
    expect(seedCourseIds).toEqual(['course-alpha', 'course-beta']);
    expect(getCourseSeedExamplesPath('course-alpha')).not.toBe(
      getCourseSeedExamplesPath('course-beta'),
    );
  });

  it('does not share evaluations across different course ids', async () => {
    mockEvaluationSubscription('course-alpha');
    const first = renderWithCourse('course-alpha', <ResultsPage />, 'results');
    await waitFor(() => {
      expect(subscribeToEvaluationsMock).toHaveBeenCalledWith(
        'course-alpha',
        expect.any(Function),
        expect.any(Function),
      );
    });
    first.unmount();

    mockEvaluationSubscription('course-beta');
    renderWithCourse('course-beta', <ResultsPage />, 'results');
    await waitFor(() => {
      expect(subscribeToEvaluationsMock).toHaveBeenCalledWith(
        'course-beta',
        expect.any(Function),
        expect.any(Function),
      );
    });

    const evaluationCourseIds = subscribeToEvaluationsMock.mock.calls.map(
      (call) => call[0],
    );
    expect(evaluationCourseIds).toEqual(['course-alpha', 'course-beta']);
    expect(getCourseEvaluationsPath('course-alpha')).not.toBe(
      getCourseEvaluationsPath('course-beta'),
    );
  });
});

describe('course-scoped create/delete helpers via hooks', () => {
  afterEach(() => {
    cleanup();
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('Evaluate saves to courses/{courseId}/evaluations', async () => {
    const { renderHook, act } = await import('@testing-library/react');
    const { useEvaluations } = await import('./useEvaluations');

    subscribeToEvaluationsMock.mockImplementation(
      (_courseId: string, onData: (evaluations: unknown[]) => void) => {
        onData([]);
        return () => undefined;
      },
    );
    createEvaluationMock.mockResolvedValue({
      id: 'eval-new',
      comparisonId: 'comparison-001',
      mostAccurate: 'rag',
      mostHelpful: 'rag',
      mostConcise: 'base',
      bestGrounded: 'rag',
      preferredModel: 'rag',
      hallucinationFlags: [],
      createdAt: '2026-01-01T00:00:00.000Z',
    });

    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <CourseProvider courseId="course-gamma">{children}</CourseProvider>
    );

    const { result } = renderHook(() => useEvaluations(), { wrapper });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    await act(async () => {
      await result.current.addEvaluation({
        id: 'temp',
        comparisonId: 'comparison-001',
        mostAccurate: 'rag',
        mostHelpful: 'rag',
        mostConcise: 'base',
        bestGrounded: 'rag',
        preferredModel: 'rag',
        hallucinationFlags: [],
        createdAt: '2026-01-01T00:00:00.000Z',
      });
    });

    expect(createEvaluationMock).toHaveBeenCalledWith(
      'course-gamma',
      expect.objectContaining({ comparisonId: 'comparison-001' }),
    );
    expect(getCourseEvaluationsPath('course-gamma')).toBe(
      'courses/course-gamma/evaluations',
    );
  });

  it('Seed Data creates under courses/{courseId}/seedExamples', async () => {
    const { renderHook, act } = await import('@testing-library/react');
    const { useSeedExamples } = await import('./useSeedExamples');

    subscribeToSeedExamplesMock.mockImplementation(
      (_courseId: string, onData: (seeds: unknown[]) => void) => {
        onData([]);
        return () => undefined;
      },
    );
    createSeedExampleMock.mockResolvedValue(undefined);

    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <CourseProvider courseId="course-delta">{children}</CourseProvider>
    );

    const { result } = renderHook(() => useSeedExamples(), { wrapper });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    await act(async () => {
      await result.current.addSeed({
        id: 'seed-temp',
        instruction: 'When does class meet?',
        response: 'Tuesday and Thursday.',
        category: 'Course Basics',
        sourceSection: 'Course Meetings',
        difficulty: 'Easy',
        directlyAnswered: true,
        origin: 'user',
      });
    });

    expect(createSeedExampleMock).toHaveBeenCalledWith(
      'course-delta',
      expect.objectContaining({ instruction: 'When does class meet?' }),
    );
    expect(getCourseSeedExamplesPath('course-delta')).toBe(
      'courses/course-delta/seedExamples',
    );
  });
});
