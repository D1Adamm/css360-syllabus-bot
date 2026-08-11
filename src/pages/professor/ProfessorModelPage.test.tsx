/** @vitest-environment jsdom */
import { cleanup, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import type { CourseModelRegistry } from '../../types';

/**
 * The professor view of a course model.
 *
 * CSS 360 has a trained adapter that nothing is currently serving. The page has
 * to say both: the model exists and is ready, and it is offline right now.
 */

const subscribeToCourseModel = vi.fn();
const fetchFineTunedHealth = vi.fn();

vi.mock('../../lib/courseModelDb', async () => {
  const actual =
    await vi.importActual<typeof import('../../lib/courseModelDb')>(
      '../../lib/courseModelDb',
    );
  return {
    ...actual,
    subscribeToCourseModel: (...args: unknown[]) => subscribeToCourseModel(...args),
  };
});

vi.mock('../../lib/adminApi', () => ({
  fetchFineTunedHealth: (...args: unknown[]) => fetchFineTunedHealth(...args),
}));

vi.mock('../../hooks/useCourseMetadata', () => ({
  useCourseMetadata: () => ({
    state: { status: 'ready' },
    metadata: { name: 'Css 360', title: 'Software engineering' },
    retry: vi.fn(),
  }),
}));

vi.mock('../../hooks/useCourseExampleCounts', () => ({
  useCourseExampleCounts: () => ({
    status: 'ready',
    counts: { total: 81, approved: 54, pending: 14, rejected: 13, edited: 0 },
  }),
}));

let courseId = 'css-360-winter-2026-a7rp';
vi.mock('../../context/CourseContext', () => ({
  useCourseId: () => courseId,
}));

import { ProfessorModelPage } from './ProfessorModelPage';

const CSS360_REGISTRY: CourseModelRegistry = {
  currentVersion: 'v1',
  versions: {
    v1: {
      version: 'v1',
      baseModel: 'meta-llama/Llama-3.2-3B-Instruct',
      trainingExampleCount: 54,
      status: 'ready',
      deployment: 'offline',
      artifactRef: 'css-360-qlora/adapter',
      createdAt: '2026-08-11T06:22:50.979Z',
    },
  },
};

/** Emits a registry for `forCourse` only; any other course sees none. */
function mockRegistry(registry: CourseModelRegistry | null, forCourse?: string) {
  subscribeToCourseModel.mockImplementation(
    (id: string, onData: (value: CourseModelRegistry | null) => void) => {
      onData(!forCourse || id === forCourse ? registry : null);
      return () => undefined;
    },
  );
}

function renderPage() {
  return render(
    <MemoryRouter>
      <ProfessorModelPage />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  courseId = 'css-360-winter-2026-a7rp';
  fetchFineTunedHealth.mockRejectedValue(new Error('service down'));
});

afterEach(() => {
  cleanup();
});

describe('ProfessorModelPage', () => {
  it('shows CSS 360’s model as ready but offline', () => {
    mockRegistry(CSS360_REGISTRY);
    renderPage();

    expect(screen.getByText(/Your course model is ready, but offline/i)).toBeInTheDocument();
    expect(screen.getByText('Ready · offline')).toBeInTheDocument();
    // The wording that denied the model existed.
    expect(screen.queryByText('Not available yet')).not.toBeInTheDocument();
  });

  it('shows what the model was trained from', () => {
    mockRegistry(CSS360_REGISTRY);
    renderPage();

    expect(screen.getByText('54 approved examples')).toBeInTheDocument();
    expect(screen.getByText('v1')).toBeInTheDocument();
  });

  it('never exposes the artifact reference or base model to a professor', () => {
    mockRegistry(CSS360_REGISTRY);
    renderPage();

    const text = document.body.textContent ?? '';
    expect(text).not.toContain('css-360-qlora/adapter');
    expect(text).not.toContain('meta-llama');
    expect(text).not.toMatch(/adapter|qlora|tillicum|slurm|gpfs/i);
  });

  it('does not infer a model from the inference service', () => {
    // A perfectly healthy shared service, but no record for this course.
    fetchFineTunedHealth.mockResolvedValue({ status: 'ok', adapterLoaded: true });
    mockRegistry(null);
    renderPage();

    expect(screen.getByText(/No course model yet/i)).toBeInTheDocument();
    expect(screen.queryByText(/ready/i)).not.toBeInTheDocument();
  });

  it('keeps registries isolated per course', () => {
    // The registry exists only for CSS 360; this page is a different course.
    courseId = 'css-490-spring-2026-cgvl';
    mockRegistry(CSS360_REGISTRY, 'css-360-winter-2026-a7rp');
    renderPage();

    expect(screen.getByText(/No course model yet/i)).toBeInTheDocument();
    expect(screen.queryByText('54 approved examples')).not.toBeInTheDocument();
  });

  it('distinguishes an unreadable registry from having no model', () => {
    subscribeToCourseModel.mockImplementation(
      (
        _id: string,
        _onData: (value: CourseModelRegistry | null) => void,
        onError?: (message: string) => void,
      ) => {
        onError?.('permission denied');
        return () => undefined;
      },
    );
    renderPage();

    expect(screen.getByRole('alert')).toHaveTextContent('Model status unavailable');
    expect(screen.queryByText(/No course model yet/i)).not.toBeInTheDocument();
  });
});
