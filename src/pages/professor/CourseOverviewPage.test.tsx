/** @vitest-environment jsdom */
import { cleanup, render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

/**
 * The professor's at-a-glance view of one course.
 *
 * The case that matters here is a course whose starter examples are still
 * being written. "0 approved" is a true count and a false impression: it
 * describes a job that has not finished, and reads as an upload that did
 * nothing.
 */

let courseId = 'css-350-winter-2026-drlb';
vi.mock('../../context/CourseContext', () => ({
  useCourseId: () => courseId,
}));

const metadataByCourse = new Map<string, CourseMetadata>();

vi.mock('../../hooks/useCourseMetadata', () => ({
  useCourseMetadata: (id: string | null) => {
    const metadata = (id && metadataByCourse.get(id)) || null;
    return {
      state: metadata
        ? ({ status: 'ready', metadata } as const)
        : ({ status: 'missing' } as const),
      metadata,
      retry: vi.fn(),
    };
  },
}));

const useCourseExampleCounts = vi.fn();
vi.mock('../../hooks/useCourseExampleCounts', () => ({
  useCourseExampleCounts: (...args: unknown[]) => useCourseExampleCounts(...args),
}));

vi.mock('../../hooks/useEvaluations', () => ({
  useEvaluations: () => ({ evaluations: [] }),
}));

/*
 * The two model records, keyed by course.
 *
 * Keyed rather than a single return value so that "does CSS 350's model leak
 * into CSS 360's overview?" is a question these tests can actually ask. The
 * real hooks re-subscribe on `courseId`; these mirror that by looking the
 * course up on every call.
 */
const modelStateByCourse = new Map<string, CourseModelState>();
const requestStateByCourse = new Map<string, CourseModelRequestState>();

vi.mock('../../hooks/useCourseModel', () => ({
  useCourseModel: (id: string | null) => ({
    state: (id && modelStateByCourse.get(id)) || ({ status: 'none' } as const),
    retry: vi.fn(),
  }),
}));

vi.mock('../../hooks/useCourseModelRequest', () => ({
  useCourseModelRequest: (id: string | null) => ({
    state: (id && requestStateByCourse.get(id)) || ({ status: 'none' } as const),
    submitting: false,
    submitError: null,
    submit: vi.fn(),
    clearSubmitError: vi.fn(),
  }),
}));

import type { CourseModelState } from '../../hooks/useCourseModel';
import type { CourseModelRequestState } from '../../hooks/useCourseModelRequest';
import type {
  CourseMetadata,
  CourseModelRegistry,
  CourseModelRequest,
  CourseModelVersion,
  StoredStarterSeedGeneration,
} from '../../types';
import { CourseOverviewPage } from './CourseOverviewPage';

const BASE_METADATA: CourseMetadata = {
  name: 'CSS 350',
  title: 'Management Principles',
  term: 'Winter 2026',
  instructorName: '',
  createdAt: '2026-08-12T09:00:00.000Z',
  syllabusStatus: 'indexed',
  syllabusFileName: 'syllabus.pdf',
  syllabusType: 'pdf',
  chunkCount: 12,
};

const NO_EXAMPLES = { total: 0, approved: 0, pending: 0, rejected: 0, edited: 0 };

function setGeneration(
  id: string,
  starterSeedGeneration: StoredStarterSeedGeneration | undefined,
) {
  metadataByCourse.set(id, {
    ...BASE_METADATA,
    ...(starterSeedGeneration ? { starterSeedGeneration } : {}),
  });
}

function renderPage() {
  return render(
    <MemoryRouter>
      <CourseOverviewPage />
    </MemoryRouter>,
  );
}

/** The Training examples row of the course-status list. */
function trainingExamplesRow(): HTMLElement {
  const term = screen.getByText('Training examples');
  return term.closest('.overview__row') as HTMLElement;
}

/** The Course model row of the course-status list. */
function courseModelRow(): HTMLElement {
  const term = screen.getByText('Course model');
  return term.closest('.overview__row') as HTMLElement;
}

const READY_VERSION: CourseModelVersion = {
  version: 'v1',
  baseModel: 'meta-llama/Llama-3.2-3B-Instruct',
  trainingExampleCount: 37,
  status: 'ready',
  deployment: 'offline',
  artifactRef: 'serving/css-350-winter-2026-drlb/v1/adapter',
  createdAt: '2026-08-27T06:48:00.000Z',
};

function registry(version: CourseModelVersion): CourseModelRegistry {
  return { currentVersion: version.version, versions: { [version.version]: version } };
}

function setModel(id: string, version: CourseModelVersion | null) {
  modelStateByCourse.set(
    id,
    version ? { status: 'ready', registry: registry(version) } : { status: 'none' },
  );
}

function request(status: CourseModelRequest['status']): CourseModelRequest {
  return {
    courseId: 'css-350-winter-2026-drlb',
    status,
    requestedAt: '2026-08-20T09:00:00.000Z',
    updatedAt: '2026-08-20T09:00:00.000Z',
    approvedExampleCount: 42,
  };
}

function setRequest(id: string, status: CourseModelRequest['status'] | null) {
  requestStateByCourse.set(
    id,
    status ? { status: 'ready', request: request(status) } : { status: 'none' },
  );
}

describe('CourseOverviewPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    courseId = 'css-350-winter-2026-drlb';
    metadataByCourse.clear();
    modelStateByCourse.clear();
    requestStateByCourse.clear();
    setGeneration(courseId, undefined);
    useCourseExampleCounts.mockReturnValue({ status: 'ready', counts: NO_EXAMPLES });
  });

  afterEach(() => {
    cleanup();
  });

  it('shows Generating instead of 0 approved while examples are being made', () => {
    setGeneration(courseId, { status: 'generating', targetCount: 50 });

    renderPage();

    const row = trainingExamplesRow();
    expect(within(row).getByText('Generating…')).toBeInTheDocument();
    expect(within(row).queryByText('0')).not.toBeInTheDocument();
    expect(row.textContent ?? '').not.toMatch(/0 approved/);
  });

  it('treats a queued job as generating', () => {
    setGeneration(courseId, { status: 'queued' });

    renderPage();

    expect(within(trainingExamplesRow()).getByText('Generating…')).toBeInTheDocument();
  });

  it('still points out examples that are already reviewable', () => {
    setGeneration(courseId, { status: 'generating' });
    useCourseExampleCounts.mockReturnValue({
      status: 'ready',
      counts: { total: 6, approved: 0, pending: 6, rejected: 0, edited: 0 },
    });

    renderPage();

    const row = trainingExamplesRow();
    expect(within(row).getByText('Generating…')).toBeInTheDocument();
    expect(row.textContent ?? '').toMatch(/6 ready to review/);
  });

  it('shows the real count once generation is ready', () => {
    setGeneration(courseId, { status: 'ready', savedCount: 48 });
    useCourseExampleCounts.mockReturnValue({
      status: 'ready',
      counts: { total: 48, approved: 12, pending: 36, rejected: 0, edited: 0 },
    });

    renderPage();

    const row = trainingExamplesRow();
    expect(row.textContent ?? '').toMatch(/12\s*approved/);
    expect(row.textContent ?? '').toMatch(/36 awaiting review/);
    expect(within(row).queryByText('Generating…')).not.toBeInTheDocument();
  });

  it('shows the real count for a course that never had a generation record', () => {
    useCourseExampleCounts.mockReturnValue({
      status: 'ready',
      counts: { total: 10, approved: 4, pending: 6, rejected: 0, edited: 0 },
    });

    renderPage();

    const row = trainingExamplesRow();
    expect(row.textContent ?? '').toMatch(/4\s*approved/);
    expect(within(row).queryByText('Generating…')).not.toBeInTheDocument();
  });

  it('says nothing about generation when the course failed to produce examples', () => {
    // The overview is not where a failure is explained; the Examples page is.
    // What it must not do is imply work is still under way.
    setGeneration(courseId, { status: 'failed', error: 'ollama timed out' });

    renderPage();

    const row = trainingExamplesRow();
    expect(within(row).queryByText('Generating…')).not.toBeInTheDocument();
    expect(document.body.textContent ?? '').not.toMatch(/ollama|timed out/i);
  });

  it('keeps one course’s generation state out of another’s overview', () => {
    setGeneration('css-360-winter-2026-a7rp', { status: 'generating' });
    setGeneration(courseId, undefined);

    renderPage();

    expect(within(trainingExamplesRow()).queryByText('Generating…')).not.toBeInTheDocument();
  });

  it('shows the same thing after a refresh, because the state is stored', () => {
    setGeneration(courseId, { status: 'generating' });

    const first = renderPage();
    expect(within(trainingExamplesRow()).getByText('Generating…')).toBeInTheDocument();

    first.unmount();
    cleanup();
    renderPage();

    expect(within(trainingExamplesRow()).getByText('Generating…')).toBeInTheDocument();
  });

  it('does not claim generation when the counts cannot be read', () => {
    useCourseExampleCounts.mockReturnValue({ status: 'unavailable' });

    renderPage();

    const row = trainingExamplesRow();
    expect(within(row).getByText('Not available right now')).toBeInTheDocument();
  });

  /* --------------------------------------------------------------------- *
   * The Course model row
   *
   * It used to be a hardcoded "Not available yet" pill: wrong for every course
   * that had requested, trained, or published anything, and the exact opposite
   * of what the Model page said one click away. It now reads the same two
   * records through the same helper that page uses.
   * --------------------------------------------------------------------- */

  describe('course model row', () => {
    it('never shows the old hardcoded text', () => {
      setModel(courseId, READY_VERSION);

      renderPage();

      expect(courseModelRow()).not.toHaveTextContent('Not available yet');
    });

    it('says nothing has been created for a course with no model or request', () => {
      renderPage();

      expect(courseModelRow()).toHaveTextContent('Not created yet');
    });

    it('shows a request that has been submitted but not started', () => {
      setRequest(courseId, 'requested');

      renderPage();

      expect(courseModelRow()).toHaveTextContent('Requested');
    });

    it('shows a request whose data is being prepared', () => {
      setRequest(courseId, 'preparing');

      renderPage();

      expect(courseModelRow()).toHaveTextContent('Being prepared');
    });

    it('shows a run that is training', () => {
      setRequest(courseId, 'training');

      renderPage();

      expect(courseModelRow()).toHaveTextContent('Training');
      expect(courseModelRow()).not.toHaveTextContent('Not available yet');
    });

    it('shows a registered model as ready, and says it is not published', () => {
      // Registered and published are different facts. A trained model that has
      // not been copied to the cluster is real, and is not answering questions.
      setModel(courseId, READY_VERSION);

      renderPage();

      expect(courseModelRow()).toHaveTextContent('Ready · not published');
    });

    it('shows a published model as published', () => {
      setModel(courseId, { ...READY_VERSION, deployment: 'online' });

      renderPage();

      expect(courseModelRow()).toHaveTextContent('Ready · published');
    });

    it('keeps showing the model while a newer version is being trained', () => {
      // The professor still has a working model. Replacing "ready" with
      // "training" here would tell them they have nothing.
      setModel(courseId, { ...READY_VERSION, deployment: 'online' });
      setRequest(courseId, 'training');

      renderPage();

      expect(courseModelRow()).toHaveTextContent('Ready · published');
      expect(courseModelRow()).not.toHaveTextContent('Training');
    });

    it('surfaces a failed request rather than calling it not created', () => {
      setRequest(courseId, 'failed');

      renderPage();

      expect(courseModelRow()).toHaveTextContent('Needs attention');
      expect(courseModelRow()).not.toHaveTextContent('Not created yet');
    });

    it('does not claim a course has no model while it is still loading', () => {
      // The bug in miniature: a guess made for a fraction of a second is
      // indistinguishable from the permanent wrong answer it replaced.
      modelStateByCourse.set(courseId, { status: 'loading' });

      renderPage();

      expect(courseModelRow()).toHaveTextContent('Checking…');
      expect(courseModelRow()).not.toHaveTextContent('Not created yet');
    });

    it('does not claim a course has no model when the registry cannot be read', () => {
      modelStateByCourse.set(courseId, {
        status: 'unavailable',
        message: 'network',
      });

      renderPage();

      expect(courseModelRow()).toHaveTextContent('Temporarily unavailable');
      expect(courseModelRow()).not.toHaveTextContent('Not created yet');
    });

    it('does not claim nothing was requested when the request cannot be read', () => {
      requestStateByCourse.set(courseId, {
        status: 'unavailable',
        message: 'network',
      });

      renderPage();

      expect(courseModelRow()).toHaveTextContent('Temporarily unavailable');
    });

    it('still shows the model when only the request record is unreadable', () => {
      // A model that exists is a fact we already have. A failed request read
      // does not take it away.
      setModel(courseId, { ...READY_VERSION, deployment: 'online' });
      requestStateByCourse.set(courseId, {
        status: 'unavailable',
        message: 'network',
      });

      renderPage();

      expect(courseModelRow()).toHaveTextContent('Ready · published');
    });

    it('keeps one course’s model state out of another’s overview', () => {
      setModel('css-350-winter-2026-drlb', { ...READY_VERSION, deployment: 'online' });
      courseId = 'css-360-winter-2026-a7rp';
      setGeneration(courseId, undefined);

      renderPage();

      expect(courseModelRow()).toHaveTextContent('Not created yet');
      expect(courseModelRow()).not.toHaveTextContent('Ready');
    });

    it('keeps one course’s request state out of another’s overview', () => {
      setRequest('css-350-winter-2026-drlb', 'training');
      courseId = 'css-360-winter-2026-a7rp';
      setGeneration(courseId, undefined);

      renderPage();

      expect(courseModelRow()).toHaveTextContent('Not created yet');
      expect(courseModelRow()).not.toHaveTextContent('Training');
    });

    it('still links to the course model page', () => {
      // Unchanged by the fix, and the reason the row is worth having.
      setModel(courseId, READY_VERSION);

      renderPage();

      const link = within(courseModelRow()).getByRole('link', { name: 'Details' });
      expect(link).toHaveAttribute(
        'href',
        `/professor/course/${courseId}/model`,
      );
    });
  });
  /* --------------------------------------------------------------------- *
   * The attention banner
   *
   * "You have N approved examples — enough for a course model" was decided by
   * the approved count alone, so it kept appearing under NEEDS YOUR ATTENTION
   * for a course whose model had been trained, registered and published weeks
   * earlier — directly above a Course model row reading "Ready · published".
   * Attention is now only claimed when the action behind it exists.
   * --------------------------------------------------------------------- */

  describe('the course-model attention item', () => {
    const ENOUGH = { total: 50, approved: 42, pending: 0, rejected: 0, edited: 0 };

    function attentionSection(): HTMLElement | null {
      return document.querySelector('.attention');
    }

    beforeEach(() => {
      useCourseExampleCounts.mockReturnValue({ status: 'ready', counts: ENOUGH });
    });

    it('asks for a model when there are enough examples and none exists', () => {
      renderPage();

      const section = attentionSection();
      expect(section).not.toBeNull();
      expect(section).toHaveTextContent(
        '42 approved examples — enough to request a course model',
      );
    });

    it('does not ask for a model when one is ready and published', () => {
      setModel(courseId, { ...READY_VERSION, deployment: 'online' });

      renderPage();

      expect(document.body.textContent ?? '').not.toMatch(/enough .* course model/);
      expect(courseModelRow()).toHaveTextContent('Ready · published');
    });

    it('hides the attention section entirely when the model is the only thing that would be in it', () => {
      setModel(courseId, { ...READY_VERSION, deployment: 'online' });

      renderPage();

      expect(attentionSection()).toBeNull();
    });

    it('does not ask for a model when one is ready but not published yet', () => {
      // Registered and unpublished is still a model. There is nothing to
      // request; publishing is an administrator's step.
      setModel(courseId, READY_VERSION);

      renderPage();

      expect(document.body.textContent ?? '').not.toMatch(/enough .* course model/);
      expect(courseModelRow()).toHaveTextContent('Ready · not published');
    });

    it('does not ask for a model while a request is already outstanding', () => {
      setRequest(courseId, 'training');

      renderPage();

      expect(document.body.textContent ?? '').not.toMatch(/enough .* course model/);
    });

    it('does not ask for a model while the registry is still being read', () => {
      modelStateByCourse.set(courseId, { status: 'loading' });

      renderPage();

      expect(document.body.textContent ?? '').not.toMatch(/enough .* course model/);
    });

    it('does not ask for a model when the registry could not be read', () => {
      modelStateByCourse.set(courseId, { status: 'unavailable', message: 'network' });

      renderPage();

      expect(document.body.textContent ?? '').not.toMatch(/enough .* course model/);
    });

    it('does not ask for a second request when the request record is unreadable', () => {
      requestStateByCourse.set(courseId, { status: 'unavailable', message: 'network' });

      renderPage();

      expect(document.body.textContent ?? '').not.toMatch(/enough .* course model/);
    });

    it('leaves the other attention items alone for a course with a published model', () => {
      // The model item is the only one this change touches.
      setModel(courseId, { ...READY_VERSION, deployment: 'online' });
      useCourseExampleCounts.mockReturnValue({
        status: 'ready',
        counts: { ...ENOUGH, pending: 3 },
      });

      renderPage();

      expect(attentionSection()).toHaveTextContent('3 examples waiting for your review');
      expect(attentionSection()).not.toHaveTextContent('course model');
    });

    it('says nothing about a model when there are too few approved examples', () => {
      useCourseExampleCounts.mockReturnValue({
        status: 'ready',
        counts: { total: 20, approved: 4, pending: 0, rejected: 0, edited: 0 },
      });

      renderPage();

      expect(document.body.textContent ?? '').not.toMatch(/enough .* course model/);
    });
  });
});
