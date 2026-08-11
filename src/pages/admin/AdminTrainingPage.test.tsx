/** @vitest-environment jsdom */
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

/**
 * Export and train/validation split.
 *
 * These behaviours moved off the professor review page, where they exposed
 * dataset mechanics and server paths to someone reviewing course content. The
 * assertions are carried over unchanged — same endpoints, same arguments, same
 * gating — only the surface they run on is different.
 */

const exportApprovedCourseSeeds = vi.fn();
const getApprovedExportStatus = vi.fn();
const prepareTrainingSplit = vi.fn();

vi.mock('../../lib/api', () => ({
  ApiError: class ApiError extends Error {
    status?: number;
    constructor(message: string, status?: number) {
      super(message);
      this.name = 'ApiError';
      this.status = status;
    }
  },
  // Approved counts now come from the examples themselves, not the export.
  listCourseSeeds: vi.fn().mockResolvedValue({
    courseId: 'css-360-winter-2026-a7rp',
    count: 2,
    firebasePath: 'courses/css-360-winter-2026-a7rp/seedExamples',
    seeds: [
      { id: 'a', question: 'q1', answer: 'a1', reviewStatus: 'approved' },
      { id: 'b', question: 'q2', answer: 'a2', reviewStatus: 'generated' },
    ],
  }),
  exportApprovedCourseSeeds: (...args: unknown[]) => exportApprovedCourseSeeds(...args),
  getApprovedExportStatus: (...args: unknown[]) => getApprovedExportStatus(...args),
  prepareTrainingSplit: (...args: unknown[]) => prepareTrainingSplit(...args),
}));

const fetchCourseModelRequest = vi.fn();

vi.mock('../../lib/courseModelRequestDb', () => ({
  fetchCourseModelRequest: (...args: unknown[]) => fetchCourseModelRequest(...args),
}));

const subscribeToCoursesMock = vi.fn();

vi.mock('../../lib/coursesDb', () => ({
  subscribeToCourses: (...args: unknown[]) => subscribeToCoursesMock(...args),
}));

import { AdminTrainingPage } from './AdminTrainingPage';

const COURSE_ID = 'css-360-winter-2026-a7rp';

function renderPage() {
  return render(<AdminTrainingPage />);
}

describe('AdminTrainingPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    subscribeToCoursesMock.mockImplementation((onData: (courses: unknown[]) => void) => {
      onData([
        {
          courseId: COURSE_ID,
          metadata: {
            name: 'CSS 360',
            title: 'Software Engineering',
            term: 'Winter 2026',
            instructorName: '',
            createdAt: '2026-01-01T00:00:00.000Z',
            syllabusStatus: 'indexed',
            syllabusFileName: 'syllabus.pdf',
            syllabusType: 'pdf',
            chunkCount: 12,
          },
        },
      ]);
      return () => undefined;
    });

    getApprovedExportStatus.mockResolvedValue({
      courseId: COURSE_ID,
      exists: false,
      exportPath: '',
      exampleCount: 0,
      sourceFile: '',
    });

    exportApprovedCourseSeeds.mockResolvedValue({
      courseId: COURSE_ID,
      summary: {
        approvedCount: 1,
        exportedCount: 1,
        validatedCount: 1,
        validationPassed: true,
        exportPath: 'data/exports/css-360-winter-2026-a7rp/approved-finetune.jsonl',
      },
    });

    prepareTrainingSplit.mockResolvedValue({
      courseId: COURSE_ID,
      summary: { trainExamples: 48, validationExamples: 6, totalExamples: 54 },
    });

    fetchCourseModelRequest.mockResolvedValue(null);
  });

  afterEach(() => {
    cleanup();
  });

  it('creates a training dataset for a course', async () => {
    renderPage();
    await screen.findByText(/CSS 360/);

    fireEvent.click(screen.getByRole('button', { name: /Create training dataset|Rebuild dataset/ }));

    await waitFor(() => {
      expect(exportApprovedCourseSeeds).toHaveBeenCalledWith(COURSE_ID);
    });

    expect(
      await screen.findByText(
        /Training dataset created from 1 approved example → data\/exports\/css-360-winter-2026-a7rp\/approved-finetune\.jsonl/,
      ),
    ).toBeInTheDocument();
  });

  it('disables the split until an approved export exists', async () => {
    renderPage();
    await screen.findByText(/CSS 360/);

    await waitFor(() => {
      expect(getApprovedExportStatus).toHaveBeenCalledWith(COURSE_ID);
    });
    expect(
      screen.getByRole('button', { name: 'Prepare training split' }),
    ).toBeDisabled();

    getApprovedExportStatus.mockResolvedValue({
      courseId: COURSE_ID,
      exists: true,
      exportPath: 'data/exports/css-360-winter-2026-a7rp/approved-finetune.jsonl',
      exampleCount: 54,
      sourceFile: 'approved-finetune.jsonl',
    });

    fireEvent.click(screen.getByRole('button', { name: /Create training dataset|Rebuild dataset/ }));
    await screen.findByText(/Training dataset created from 1 approved example/);

    expect(
      screen.getByRole('button', { name: 'Prepare training split' }),
    ).not.toBeDisabled();
  });

  it('prepares a training split once an approved export exists', async () => {
    getApprovedExportStatus.mockResolvedValue({
      courseId: COURSE_ID,
      exists: true,
      exportPath: 'data/exports/css-360-winter-2026-a7rp/approved-finetune.jsonl',
      exampleCount: 54,
      sourceFile: 'approved-finetune.jsonl',
    });

    renderPage();
    await screen.findByText(/CSS 360/);

    await waitFor(() => {
      expect(
        screen.getByRole('button', { name: 'Prepare training split' }),
      ).not.toBeDisabled();
    });

    fireEvent.click(screen.getByRole('button', { name: 'Prepare training split' }));

    await waitFor(() => {
      expect(prepareTrainingSplit).toHaveBeenCalledWith(COURSE_ID);
    });
    expect(
      await screen.findByText('Prepared split: 48 train, 6 validation'),
    ).toBeInTheDocument();
  });

  it('shows an outstanding model request with its course, count, time and status', async () => {
    fetchCourseModelRequest.mockResolvedValue({
      courseId: COURSE_ID,
      status: 'requested',
      requestedAt: '2026-08-11T10:00:00.000Z',
      updatedAt: '2026-08-11T10:00:00.000Z',
      approvedExampleCount: 54,
    });

    renderPage();

    const requests = await screen.findByRole('list', { name: 'Model requests' });
    expect(within(requests).getByText(/CSS 360/)).toBeInTheDocument();
    expect(within(requests).getByText(new RegExp(COURSE_ID))).toBeInTheDocument();
    expect(
      within(requests).getByText(/54 approved examples at request/),
    ).toBeInTheDocument();
    expect(within(requests).getByText('requested')).toBeInTheDocument();
  });

  it('asks each course for its own request', async () => {
    renderPage();

    await waitFor(() => {
      expect(fetchCourseModelRequest).toHaveBeenCalledWith(COURSE_ID);
    });
  });

  it('says so when no course has requested a model', async () => {
    renderPage();

    expect(await screen.findByText('No model requests')).toBeInTheDocument();
  });

  it('offers no training-launch control alongside a request', async () => {
    fetchCourseModelRequest.mockResolvedValue({
      courseId: COURSE_ID,
      status: 'requested',
      requestedAt: '2026-08-11T10:00:00.000Z',
      updatedAt: '2026-08-11T10:00:00.000Z',
      approvedExampleCount: 54,
    });

    renderPage();

    const requests = await screen.findByRole('list', { name: 'Model requests' });
    // Acting on a request is still manual; nothing here submits a job.
    expect(within(requests).queryByRole('button')).toBeNull();
  });

  it('shows an admin the recorded failure detail a professor never sees', async () => {
    fetchCourseModelRequest.mockResolvedValue({
      courseId: COURSE_ID,
      status: 'failed',
      requestedAt: '2026-08-11T10:00:00.000Z',
      updatedAt: '2026-08-11T11:00:00.000Z',
      approvedExampleCount: 54,
      failureMessage: 'run exited non-zero',
    });

    renderPage();

    expect(await screen.findByText('run exited non-zero')).toBeInTheDocument();
  });
});
