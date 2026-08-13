/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
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
const subscribeToCourseModelRequest = vi.fn();
const createCourseModelRequest = vi.fn();
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

vi.mock('../../lib/courseModelRequestDb', async () => {
  const actual = await vi.importActual<
    typeof import('../../lib/courseModelRequestDb')
  >('../../lib/courseModelRequestDb');
  return {
    ...actual,
    subscribeToCourseModelRequest: (...args: unknown[]) =>
      subscribeToCourseModelRequest(...args),
    createCourseModelRequest: (...args: unknown[]) =>
      createCourseModelRequest(...args),
  };
});

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

import type { CourseModelRequest } from '../../types';
import { ProfessorModelPage } from './ProfessorModelPage';

/** Emits a request for `forCourse` only; any other course sees none. */
function mockRequest(request: CourseModelRequest | null, forCourse?: string) {
  subscribeToCourseModelRequest.mockImplementation(
    (id: string, onData: (value: CourseModelRequest | null) => void) => {
      onData(!forCourse || id === forCourse ? request : null);
      return () => undefined;
    },
  );
}

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
  mockRequest(null);
  createCourseModelRequest.mockResolvedValue(undefined);
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

  it('offers no first-model request for a course that already has one', () => {
    // CSS 360 has v1. Offering "Request course model" here would be nonsense.
    mockRegistry(CSS360_REGISTRY);
    renderPage();

    expect(
      screen.queryByRole('button', { name: /Request course model/i }),
    ).not.toBeInTheDocument();
  });
});

describe('ProfessorModelPage requests', () => {
  it('offers a request when there is no model and enough approved examples', () => {
    mockRegistry(null);
    renderPage();

    expect(
      screen.getByRole('button', { name: /Request course model/i }),
    ).toBeInTheDocument();
  });

  it('persists the request with the approved count', async () => {
    mockRegistry(null);
    renderPage();

    fireEvent.click(screen.getByRole('button', { name: /Request course model/i }));

    await waitFor(() => {
      expect(createCourseModelRequest).toHaveBeenCalledWith(
        'css-360-winter-2026-a7rp',
        54,
      );
    });
  });

  it('shows the request status instead of the button once one exists', () => {
    mockRegistry(null);
    mockRequest({
      courseId: 'css-360-winter-2026-a7rp',
      status: 'requested',
      requestedAt: '2026-08-11T10:00:00.000Z',
      updatedAt: '2026-08-11T10:00:00.000Z',
      approvedExampleCount: 54,
    });
    renderPage();

    expect(
      screen.getByText(/Your course model has been requested/i),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: /Request course model/i }),
    ).not.toBeInTheDocument();
  });

  it('gives a professor simple language for each stage', () => {
    // Requested / Being prepared / Training — no pipeline vocabulary.
    const expected = {
      requested: /has been requested/i,
      preparing: /is being prepared/i,
      training: /is training/i,
    } as const;

    for (const [status, pattern] of Object.entries(expected)) {
      mockRegistry(null);
      mockRequest({
        courseId: 'css-360-winter-2026-a7rp',
        status: status as 'requested' | 'preparing' | 'training',
        requestedAt: '2026-08-11T10:00:00.000Z',
        updatedAt: '2026-08-11T10:00:00.000Z',
        approvedExampleCount: 54,
      });
      renderPage();

      expect(screen.getByText(pattern)).toBeInTheDocument();
      cleanup();
    }
  });

  it('never shows a recorded failure message to a professor', () => {
    mockRegistry(null);
    mockRequest({
      courseId: 'css-360-winter-2026-a7rp',
      status: 'failed',
      requestedAt: '2026-08-11T10:00:00.000Z',
      updatedAt: '2026-08-11T11:00:00.000Z',
      approvedExampleCount: 54,
      failureMessage: 'sbatch job 91231 failed on gpu node n2145',
    });
    renderPage();

    const text = document.body.textContent ?? '';
    expect(text).not.toContain('sbatch');
    expect(text).not.toContain('n2145');
    expect(screen.getByText(/didn't complete/i)).toBeInTheDocument();
  });

  it('keeps requests isolated per course', () => {
    courseId = 'css-490-spring-2026-cgvl';
    mockRegistry(null);
    mockRequest(
      {
        courseId: 'css-360-winter-2026-a7rp',
        status: 'training',
        requestedAt: '2026-08-11T10:00:00.000Z',
        updatedAt: '2026-08-11T10:00:00.000Z',
        approvedExampleCount: 54,
      },
      'css-360-winter-2026-a7rp',
    );
    renderPage();

    // Another course's in-flight request must not appear here.
    expect(screen.queryByText(/being prepared/i)).not.toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /Request course model/i }),
    ).toBeInTheDocument();
  });

  it('exposes no infrastructure vocabulary in any request state', () => {
    for (const status of ['requested', 'preparing', 'training', 'failed'] as const) {
      mockRegistry(null);
      mockRequest({
        courseId: 'css-360-winter-2026-a7rp',
        status,
        requestedAt: '2026-08-11T10:00:00.000Z',
        updatedAt: '2026-08-11T10:00:00.000Z',
        approvedExampleCount: 54,
      });
      renderPage();

      expect(document.body.textContent ?? '').not.toMatch(
        /tillicum|slurm|sbatch|ssh|duo|gpu|node|adapter|qlora|hugging ?face|token|http|\.edu/i,
      );
      cleanup();
    }
  });

  it('offers no preparation control to a professor', () => {
    mockRegistry(null);
    mockRequest({
      courseId: 'css-360-winter-2026-a7rp',
      status: 'preparing',
      requestedAt: '2026-08-11T10:00:00.000Z',
      updatedAt: '2026-08-11T12:00:00.000Z',
      approvedExampleCount: 54,
      preparation: {
        preparedAt: '2026-08-11T12:00:00.000Z',
        sourceApprovedExampleCount: 54,
        datasetRef: 'exports/css-360-winter-2026-a7rp',
        trainExamples: 48,
        validationExamples: 6,
        splitSeed: 360,
      },
    });
    renderPage();

    // Preparing training data is an administrator action.
    expect(
      screen.queryByRole('button', { name: /Prepare training data/i }),
    ).not.toBeInTheDocument();
  });

  it('shows no dataset internals from a prepared request', () => {
    mockRegistry(null);
    mockRequest({
      courseId: 'css-360-winter-2026-a7rp',
      status: 'preparing',
      requestedAt: '2026-08-11T10:00:00.000Z',
      updatedAt: '2026-08-11T12:00:00.000Z',
      approvedExampleCount: 54,
      preparation: {
        preparedAt: '2026-08-11T12:00:00.000Z',
        sourceApprovedExampleCount: 54,
        datasetRef: 'exports/css-360-winter-2026-a7rp',
        trainExamples: 48,
        validationExamples: 6,
        splitSeed: 360,
      },
      preparationError: 'sbatch rejected the job on gpu node n2145',
    });
    renderPage();

    const text = document.body.textContent ?? '';
    expect(text).not.toMatch(/exports\/|train\b.*validation|split seed|jsonl/i);
    expect(text).not.toMatch(/sbatch|n2145|gpu/i);
    // The professor sees only that it is under way.
    expect(screen.getByText(/is being prepared/i)).toBeInTheDocument();
  });

  it('shows a training request without any job or cluster detail', () => {
    mockRegistry(null);
    mockRequest({
      courseId: 'css-360-winter-2026-a7rp',
      status: 'training',
      requestedAt: '2026-08-11T10:00:00.000Z',
      updatedAt: '2026-08-11T13:00:00.000Z',
      approvedExampleCount: 54,
      preparation: {
        preparedAt: '2026-08-11T12:00:00.000Z',
        sourceApprovedExampleCount: 54,
        datasetRef: 'exports/css-360-winter-2026-a7rp',
        trainExamples: 48,
        validationExamples: 6,
      },
      training: {
        jobId: '9182736',
        mode: 'full',
        submittedAt: '2026-08-11T13:00:00.000Z',
        datasetRef: 'exports/css-360-winter-2026-a7rp',
        trainExamples: 48,
        validationExamples: 6,
      },
      launchError: 'ssh: connect to host tillicum.hyak.uw.edu port 22: refused',
    });
    renderPage();

    const text = document.body.textContent ?? '';

    // Simple language only.
    expect(screen.getByText(/is training/i)).toBeInTheDocument();

    // No job id, host, path, or account detail anywhere.
    expect(text).not.toContain('9182736');
    expect(text).not.toMatch(/tillicum|hyak|gpfs|slurm|sbatch|ssh|duo/i);
    expect(text).not.toMatch(/exports\/|jsonl|adapter|hugging/i);
    // No Start training control for a professor.
    expect(
      screen.queryByRole('button', { name: /Start training/i }),
    ).not.toBeInTheDocument();
  });

  /*
   * A queued training run.
   *
   * The request carries a pointer to it, and that pointer is the only thing
   * about the queue that touches a professor-facing record. None of it — the
   * run id, the queue, the machine that will pick it up — means anything to a
   * professor, so none of it is shown. They see the same "being prepared" they
   * saw before it was queued, which is true: nothing is training yet.
   */
  it('says nothing about a queued run beyond the stage it is at', () => {
    mockRegistry(null);
    mockRequest({
      courseId: 'css-360-winter-2026-a7rp',
      status: 'preparing',
      requestedAt: '2026-08-11T10:00:00.000Z',
      updatedAt: '2026-08-12T12:00:00.000Z',
      approvedExampleCount: 54,
      preparation: {
        preparedAt: '2026-08-11T12:00:00.000Z',
        sourceApprovedExampleCount: 54,
        datasetRef: 'exports/css-360-winter-2026-a7rp',
        trainExamples: 48,
        validationExamples: 6,
      },
      currentRunId: 'run-20260812t120000z-0a1b2c',
    });
    renderPage();

    const text = document.body.textContent ?? '';

    expect(screen.getByText('Being prepared')).toBeInTheDocument();
    expect(screen.getByText(/is being prepared/i)).toBeInTheDocument();

    // The run id, and everything the run exists to describe, stays internal.
    expect(text).not.toContain('run-20260812t120000z-0a1b2c');
    expect(text).not.toMatch(/\brun\b|queue|claim|lease|attempt/i);
    expect(text).not.toMatch(
      /courses\/|trainingRuns|firebase|tillicum|hyak|slurm|sbatch|ssh|duo|gpu/i,
    );
  });

  it('shows a professor only the three stages that mean something to them', () => {
    for (const status of ['requested', 'preparing', 'training'] as const) {
      mockRegistry(null);
      mockRequest({
        courseId: 'css-360-winter-2026-a7rp',
        status,
        requestedAt: '2026-08-11T10:00:00.000Z',
        updatedAt: '2026-08-12T12:00:00.000Z',
        approvedExampleCount: 54,
        currentRunId: 'run-20260812t120000z-0a1b2c',
      });
      renderPage();

      const label = { requested: 'Requested', preparing: 'Being prepared', training: 'Training' }[
        status
      ];
      expect(screen.getByText(label)).toBeInTheDocument();
      expect(document.body.textContent ?? '').not.toContain(
        'run-20260812t120000z-0a1b2c',
      );
      cleanup();
    }
  });
});
