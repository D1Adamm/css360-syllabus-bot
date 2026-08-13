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

import type { CourseMetadata, StoredStarterSeedGeneration } from '../../types';
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

describe('CourseOverviewPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    courseId = 'css-350-winter-2026-drlb';
    metadataByCourse.clear();
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
});
