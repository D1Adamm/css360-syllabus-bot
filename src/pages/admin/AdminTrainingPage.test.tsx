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
const prepareTrainingDataForRequest = vi.fn();
const queueTrainingForRequest = vi.fn();
const fetchCourseTrainingRuns = vi.fn();

vi.mock('../../lib/courseModelRequestDb', () => ({
  fetchCourseModelRequest: (...args: unknown[]) => fetchCourseModelRequest(...args),
}));

vi.mock('../../lib/queueTraining', async () => {
  const actual = await vi.importActual<typeof import('../../lib/queueTraining')>(
    '../../lib/queueTraining',
  );
  return {
    ...actual,
    queueTrainingForRequest: (...args: unknown[]) => queueTrainingForRequest(...args),
  };
});

vi.mock('../../lib/trainingRunDb', async () => {
  const actual = await vi.importActual<typeof import('../../lib/trainingRunDb')>(
    '../../lib/trainingRunDb',
  );
  return {
    ...actual,
    fetchCourseTrainingRuns: (...args: unknown[]) => fetchCourseTrainingRuns(...args),
  };
});

vi.mock('../../lib/prepareTrainingData', async () => {
  const actual = await vi.importActual<typeof import('../../lib/prepareTrainingData')>(
    '../../lib/prepareTrainingData',
  );
  return {
    ...actual,
    prepareTrainingDataForRequest: (...args: unknown[]) =>
      prepareTrainingDataForRequest(...args),
  };
});

const subscribeToCoursesMock = vi.fn();

vi.mock('../../lib/coursesDb', () => ({
  subscribeToCourses: (...args: unknown[]) => subscribeToCoursesMock(...args),
}));

import { InsufficientApprovedExamplesError } from '../../lib/prepareTrainingData';
import { DuplicateTrainingRunError } from '../../lib/trainingRunDb';
import type { TrainingRun } from '../../types';
import { AdminTrainingPage } from './AdminTrainingPage';

const COURSE_ID = 'css-360-winter-2026-a7rp';

const PREPARED_REQUEST = {
  courseId: COURSE_ID,
  status: 'preparing' as const,
  requestedAt: '2026-08-11T10:00:00.000Z',
  updatedAt: '2026-08-11T12:00:00.000Z',
  approvedExampleCount: 42,
  preparation: {
    preparedAt: '2026-08-11T12:00:00.000Z',
    sourceApprovedExampleCount: 42,
    datasetRef: `exports/${COURSE_ID}`,
    trainExamples: 38,
    validationExamples: 4,
    splitSeed: 360,
  },
};

const QUEUED_RUN: TrainingRun = {
  runId: 'run-20260812t120000z-0a1b2c',
  courseId: COURSE_ID,
  mode: 'full',
  state: 'queued',
  enqueuedAt: '2026-08-12T12:00:00.000Z',
  updatedAt: '2026-08-12T12:00:00.000Z',
  datasetRef: `exports/${COURSE_ID}`,
  approvedExampleCount: 42,
  trainExamples: 38,
  validationExamples: 4,
  attempt: 0,
};

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
    fetchCourseTrainingRuns.mockResolvedValue([]);
    queueTrainingForRequest.mockResolvedValue({ run: QUEUED_RUN });
    prepareTrainingDataForRequest.mockResolvedValue({
      preparation: {
        preparedAt: '2026-08-11T12:00:00.000Z',
        sourceApprovedExampleCount: 42,
        datasetRef: `exports/${COURSE_ID}`,
        trainExamples: 38,
        validationExamples: 4,
        splitSeed: 360,
      },
    });
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

  it('offers no job-submission control alongside a request', async () => {
    fetchCourseModelRequest.mockResolvedValue({
      courseId: COURSE_ID,
      status: 'requested',
      requestedAt: '2026-08-11T10:00:00.000Z',
      updatedAt: '2026-08-11T10:00:00.000Z',
      approvedExampleCount: 54,
    });

    renderPage();

    const requests = await screen.findByRole('list', { name: 'Model requests' });
    // Preparing data is offered; submitting a training job is not.
    expect(
      within(requests).getByRole('button', { name: /Prepare training data/i }),
    ).toBeInTheDocument();
    for (const button of within(requests).getAllByRole('button')) {
      expect(button.textContent ?? '').not.toMatch(
        /start|submit|launch|run training|train now/i,
      );
    }
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

  it('prepares training data for that exact course', async () => {
    fetchCourseModelRequest.mockResolvedValue({
      courseId: COURSE_ID,
      status: 'requested',
      requestedAt: '2026-08-11T10:00:00.000Z',
      updatedAt: '2026-08-11T10:00:00.000Z',
      approvedExampleCount: 42,
    });

    renderPage();

    fireEvent.click(
      await screen.findByRole('button', { name: /Prepare training data/i }),
    );

    await waitFor(() => {
      expect(prepareTrainingDataForRequest).toHaveBeenCalledWith(COURSE_ID);
    });
    expect(
      await screen.findByText(/Prepared 38 train \/ 4 validation from 42 approved/),
    ).toBeInTheDocument();
  });

  it('shows whether training data has been prepared', async () => {
    fetchCourseModelRequest.mockResolvedValue({
      courseId: COURSE_ID,
      status: 'preparing',
      requestedAt: '2026-08-11T10:00:00.000Z',
      updatedAt: '2026-08-11T12:00:00.000Z',
      approvedExampleCount: 42,
      preparation: {
        preparedAt: '2026-08-11T12:00:00.000Z',
        sourceApprovedExampleCount: 42,
        datasetRef: `exports/${COURSE_ID}`,
        trainExamples: 38,
        validationExamples: 4,
        splitSeed: 360,
      },
    });

    renderPage();

    const requests = await screen.findByRole('list', { name: 'Model requests' });
    expect(within(requests).getByText(/prepared/)).toBeInTheDocument();
    expect(within(requests).getByText(/38 train \/ 4 validation/)).toBeInTheDocument();
    expect(within(requests).getByText(`exports/${COURSE_ID}`)).toBeInTheDocument();
    // Re-preparing is allowed once data exists.
    expect(
      within(requests).getByRole('button', { name: /Re-prepare training data/i }),
    ).toBeInTheDocument();
  });

  it('surfaces a refusal when the course no longer has enough approved examples', async () => {
    fetchCourseModelRequest.mockResolvedValue({
      courseId: COURSE_ID,
      status: 'requested',
      requestedAt: '2026-08-11T10:00:00.000Z',
      updatedAt: '2026-08-11T10:00:00.000Z',
      approvedExampleCount: 42,
    });
    prepareTrainingDataForRequest.mockRejectedValue(
      new InsufficientApprovedExamplesError(12, 30),
    );

    renderPage();

    fireEvent.click(
      await screen.findByRole('button', { name: /Prepare training data/i }),
    );

    expect(await screen.findByText(/12 approved examples/)).toBeInTheDocument();
  });

  it('shows the admin why the last attempt failed', async () => {
    fetchCourseModelRequest.mockResolvedValue({
      courseId: COURSE_ID,
      status: 'requested',
      requestedAt: '2026-08-11T10:00:00.000Z',
      updatedAt: '2026-08-11T11:00:00.000Z',
      approvedExampleCount: 42,
      preparationError: 'no approved export found',
    });

    renderPage();

    expect(
      await screen.findByText(/Last attempt: no approved export found/),
    ).toBeInTheDocument();
  });

  it('offers Queue training only once data is prepared', async () => {
    fetchCourseModelRequest.mockResolvedValue({
      courseId: COURSE_ID,
      status: 'requested',
      requestedAt: '2026-08-11T10:00:00.000Z',
      updatedAt: '2026-08-11T10:00:00.000Z',
      approvedExampleCount: 42,
    });

    renderPage();
    await screen.findByRole('list', { name: 'Model requests' });

    expect(screen.queryByRole('button', { name: /Queue training/i })).toBeNull();
  });

  it('queues a run for the prepared course', async () => {
    fetchCourseModelRequest.mockResolvedValue(PREPARED_REQUEST);

    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: /Queue training/i }));

    await waitFor(() => {
      expect(queueTrainingForRequest).toHaveBeenCalledWith(
        COURSE_ID,
        expect.objectContaining({ courseId: COURSE_ID, status: 'preparing' }),
      );
    });
    expect(
      await screen.findByText(new RegExp(`Queued full run ${QUEUED_RUN.runId}`)),
    ).toBeInTheDocument();
  });

  it('no longer reaches the backend launch boundary', async () => {
    fetchCourseModelRequest.mockResolvedValue(PREPARED_REQUEST);

    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: /Queue training/i }));

    await waitFor(() => {
      expect(queueTrainingForRequest).toHaveBeenCalled();
    });
    // Enqueueing is the whole of it: no submission endpoint, no job id.
    const requests = await screen.findByRole('list', { name: 'Model requests' });
    expect(within(requests).queryByRole('button', { name: /Start training/i })).toBeNull();
    expect(requests.textContent ?? '').not.toMatch(/job \d|sbatch|ssh|slurm/i);
  });

  it('shows a queued run with its state and counts', async () => {
    fetchCourseModelRequest.mockResolvedValue({
      ...PREPARED_REQUEST,
      currentRunId: QUEUED_RUN.runId,
    });
    fetchCourseTrainingRuns.mockResolvedValue([QUEUED_RUN]);

    renderPage();

    const requests = await screen.findByRole('list', { name: 'Model requests' });
    expect(within(requests).getByText(QUEUED_RUN.runId)).toBeInTheDocument();
    expect(within(requests).getByText(/run queued/)).toBeInTheDocument();
    expect(
      within(requests).getByText(/38 train \/ 4 validation · queued/),
    ).toBeInTheDocument();
  });

  it('shows who holds a claimed run and until when', async () => {
    fetchCourseModelRequest.mockResolvedValue(PREPARED_REQUEST);
    fetchCourseTrainingRuns.mockResolvedValue([
      {
        ...QUEUED_RUN,
        state: 'claimed',
        attempt: 1,
        claim: {
          owner: 'alice@tillicum',
          claimedAt: '2026-08-12T12:05:00.000Z',
          expiresAt: '2026-08-12T12:20:00.000Z',
        },
      },
    ]);

    renderPage();

    const requests = await screen.findByRole('list', { name: 'Model requests' });
    expect(within(requests).getByText(/held by alice@tillicum/)).toBeInTheDocument();
    expect(within(requests).getByText(/attempt 1/)).toBeInTheDocument();
  });

  it('offers no second queue control while a run is outstanding', async () => {
    fetchCourseModelRequest.mockResolvedValue(PREPARED_REQUEST);
    fetchCourseTrainingRuns.mockResolvedValue([QUEUED_RUN]);

    renderPage();

    const requests = await screen.findByRole('list', { name: 'Model requests' });
    expect(within(requests).queryByRole('button', { name: /Queue training/i })).toBeNull();
  });

  it('offers a fresh run once the last one finished', async () => {
    fetchCourseModelRequest.mockResolvedValue(PREPARED_REQUEST);
    fetchCourseTrainingRuns.mockResolvedValue([{ ...QUEUED_RUN, state: 'failed' }]);

    renderPage();

    expect(
      await screen.findByRole('button', { name: /Queue training/i }),
    ).toBeInTheDocument();
  });

  it('reads each course queue separately from its request', async () => {
    fetchCourseModelRequest.mockResolvedValue(PREPARED_REQUEST);

    renderPage();

    await waitFor(() => {
      expect(fetchCourseTrainingRuns).toHaveBeenCalledWith(COURSE_ID);
    });
  });

  it('shows a submitted job and offers no second start', async () => {
    fetchCourseModelRequest.mockResolvedValue({
      courseId: COURSE_ID,
      status: 'training',
      requestedAt: '2026-08-11T10:00:00.000Z',
      updatedAt: '2026-08-11T13:00:00.000Z',
      approvedExampleCount: 42,
      preparation: {
        preparedAt: '2026-08-11T12:00:00.000Z',
        sourceApprovedExampleCount: 42,
        datasetRef: `exports/${COURSE_ID}`,
        trainExamples: 38,
        validationExamples: 4,
      },
      training: {
        jobId: '9182736',
        mode: 'full',
        submittedAt: '2026-08-11T13:00:00.000Z',
        datasetRef: `exports/${COURSE_ID}`,
        trainExamples: 38,
        validationExamples: 4,
      },
    });

    renderPage();

    const requests = await screen.findByRole('list', { name: 'Model requests' });
    expect(within(requests).getByText('9182736')).toBeInTheDocument();
    expect(within(requests).getByText('training')).toBeInTheDocument();
    // A course already training is not offered a run.
    expect(within(requests).queryByRole('button', { name: /Queue training/i })).toBeNull();
  });

  it('surfaces a refused duplicate to the admin', async () => {
    fetchCourseModelRequest.mockResolvedValue(PREPARED_REQUEST);
    queueTrainingForRequest.mockRejectedValue(new DuplicateTrainingRunError());

    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: /Queue training/i }));

    expect(
      await screen.findByText(/already queued or under way for this course/),
    ).toBeInTheDocument();
  });

  it('surfaces a queueing failure to the admin', async () => {
    fetchCourseModelRequest.mockResolvedValue(PREPARED_REQUEST);
    queueTrainingForRequest.mockRejectedValue(
      new Error('Firebase is unavailable right now.'),
    );

    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: /Queue training/i }));

    expect(
      await screen.findByText(/Firebase is unavailable right now/),
    ).toBeInTheDocument();
  });

  it('shows the admin why the last launch attempt failed', async () => {
    fetchCourseModelRequest.mockResolvedValue({
      courseId: COURSE_ID,
      status: 'preparing',
      requestedAt: '2026-08-11T10:00:00.000Z',
      updatedAt: '2026-08-11T13:00:00.000Z',
      approvedExampleCount: 42,
      preparation: {
        preparedAt: '2026-08-11T12:00:00.000Z',
        sourceApprovedExampleCount: 42,
        datasetRef: `exports/${COURSE_ID}`,
        trainExamples: 38,
        validationExamples: 4,
      },
      launchError: 'rsync: connection closed',
    });

    renderPage();

    expect(
      await screen.findByText(/Last launch attempt: rsync: connection closed/),
    ).toBeInTheDocument();
  });
});
