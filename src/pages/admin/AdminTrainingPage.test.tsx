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
const queueNewVersionForRequest = vi.fn();
const retryTrainingForRequest = vi.fn();
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
    queueNewVersionForRequest: (...args: unknown[]) =>
      queueNewVersionForRequest(...args),
    retryTrainingForRequest: (...args: unknown[]) => retryTrainingForRequest(...args),
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

/**
 * The stuck shape this feature exists for: a submitted run whose cluster job
 * finished without a completion callback ever reaching PostgreSQL.
 */
const STALE_SUBMITTED_RUN: TrainingRun = {
  runId: 'run-20260823t064333z-3c94f0',
  courseId: COURSE_ID,
  mode: 'full',
  state: 'submitted',
  enqueuedAt: '2026-08-23T06:40:00.000Z',
  updatedAt: '2026-08-23T06:43:33.000Z',
  datasetRef: `exports/${COURSE_ID}`,
  approvedExampleCount: 42,
  trainExamples: 37,
  validationExamples: 5,
  attempt: 1,
  jobId: '253552',
};

const STALE_REQUEST = {
  courseId: COURSE_ID,
  status: 'training' as const,
  requestedAt: '2026-08-20T09:00:00.000Z',
  updatedAt: '2026-08-23T06:43:33.000Z',
  approvedExampleCount: 42,
  currentRunId: STALE_SUBMITTED_RUN.runId,
  preparation: {
    preparedAt: '2026-08-23T06:40:00.000Z',
    sourceApprovedExampleCount: 42,
    datasetRef: `exports/${COURSE_ID}`,
    trainExamples: 37,
    validationExamples: 5,
    splitSeed: 350,
  },
  training: {
    jobId: '253552',
    mode: 'full',
    submittedAt: '2026-08-23T06:43:33.000Z',
    datasetRef: `exports/${COURSE_ID}`,
    trainExamples: 37,
    validationExamples: 5,
  },
};

const REPLACEMENT_RUN: TrainingRun = {
  runId: 'run-20260826t170000z-9f0e1d',
  courseId: COURSE_ID,
  mode: 'full',
  state: 'queued',
  enqueuedAt: '2026-08-26T17:00:00.000Z',
  updatedAt: '2026-08-26T17:00:00.000Z',
  datasetRef: `exports/${COURSE_ID}`,
  approvedExampleCount: 42,
  trainExamples: 37,
  validationExamples: 5,
  attempt: 0,
};

const SUPERSEDED_RUN: TrainingRun = {
  ...STALE_SUBMITTED_RUN,
  state: 'failed',
  updatedAt: '2026-08-26T17:00:00.000Z',
  error: 'Superseded by admin retry',
};

/*
 * The state CSS 350 was actually in when the gap showed up: a request that has
 * gone all the way to `ready`, a succeeded run, a registered v1, and a prepared
 * dataset still sitting on the backend.
 *
 * Every control on the page was gated on `preparing`, so this course had no way
 * to train again — the only visible actions were Rebuild dataset and Prepare
 * training split, neither of which starts a run.
 */
const READY_SUCCEEDED_RUN: TrainingRun = {
  runId: 'run-20260827t064701z-1cf650',
  courseId: COURSE_ID,
  mode: 'full',
  state: 'succeeded',
  enqueuedAt: '2026-08-27T06:40:00.000Z',
  updatedAt: '2026-08-27T06:48:00.000Z',
  datasetRef: `exports/${COURSE_ID}`,
  approvedExampleCount: 42,
  trainExamples: 37,
  validationExamples: 5,
  attempt: 1,
  jobId: '264787',
};

const READY_REQUEST = {
  courseId: COURSE_ID,
  status: 'ready' as const,
  requestedAt: '2026-08-20T09:00:00.000Z',
  updatedAt: '2026-08-27T06:48:00.000Z',
  approvedExampleCount: 42,
  currentRunId: READY_SUCCEEDED_RUN.runId,
  preparation: {
    preparedAt: '2026-08-27T06:40:00.000Z',
    sourceApprovedExampleCount: 42,
    datasetRef: `exports/${COURSE_ID}`,
    trainExamples: 37,
    validationExamples: 5,
    splitSeed: 360,
  },
};

const NEW_VERSION_RUN: TrainingRun = {
  runId: 'run-20260902t080000z-abcdef',
  courseId: COURSE_ID,
  mode: 'full',
  state: 'queued',
  enqueuedAt: '2026-09-02T08:00:00.000Z',
  updatedAt: '2026-09-02T08:00:00.000Z',
  datasetRef: `exports/${COURSE_ID}`,
  approvedExampleCount: 42,
  trainExamples: 37,
  validationExamples: 5,
  attempt: 0,
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
    queueNewVersionForRequest.mockResolvedValue({ run: NEW_VERSION_RUN });
    retryTrainingForRequest.mockResolvedValue({
      run: REPLACEMENT_RUN,
      supersededRunId: STALE_SUBMITTED_RUN.runId,
    });
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
      new Error('The database is unavailable right now.'),
    );

    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: /Queue training/i }));

    expect(
      await screen.findByText(/The database is unavailable right now/),
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
  /* --------------------------------------------------------------------- *
   * Retrying a stale run
   *
   * The recovery path for a run PostgreSQL still believes is active when it
   * is not — a cluster job that finished without its completion callback ever
   * landing. Before this, the request stayed `training` forever and the
   * one-active-run guard correctly refused a replacement, so there was no way
   * out of the state that did not involve raw SQL.
   * --------------------------------------------------------------------- */

  async function openRetryConfirmation() {
    fireEvent.click(await screen.findByRole('button', { name: /Retry training/i }));
    return screen.findByRole('alertdialog');
  }

  it('offers Retry training for a stale submitted run', async () => {
    fetchCourseModelRequest.mockResolvedValue(STALE_REQUEST);
    fetchCourseTrainingRuns.mockResolvedValue([STALE_SUBMITTED_RUN]);

    renderPage();

    expect(
      await screen.findByRole('button', { name: /Retry training/i }),
    ).toBeInTheDocument();
  });

  it('offers Retry training for a failed run', async () => {
    fetchCourseModelRequest.mockResolvedValue({
      ...STALE_REQUEST,
      status: 'failed' as const,
    });
    fetchCourseTrainingRuns.mockResolvedValue([
      { ...STALE_SUBMITTED_RUN, state: 'failed' as const, error: 'CUDA OOM' },
    ]);

    renderPage();

    expect(
      await screen.findByRole('button', { name: /Retry training/i }),
    ).toBeInTheDocument();
  });

  it('offers no Retry training for a job that reported minutes ago', async () => {
    /*
     * The dangerous case: the run looks stuck, but its Slurm job is alive and
     * about to report. Retiring it would leave two jobs writing the same
     * adapter, so neither side offers it.
     */
    fetchCourseModelRequest.mockResolvedValue(STALE_REQUEST);
    fetchCourseTrainingRuns.mockResolvedValue([
      { ...STALE_SUBMITTED_RUN, updatedAt: new Date(Date.now() - 300_000).toISOString() },
    ]);

    renderPage();

    await screen.findByRole('list', { name: 'Model requests' });
    expect(screen.queryByRole('button', { name: /Retry training/i })).toBeNull();
  });

  it('offers no Retry training for a healthy queued run', async () => {
    fetchCourseModelRequest.mockResolvedValue({
      ...PREPARED_REQUEST,
      currentRunId: QUEUED_RUN.runId,
    });
    fetchCourseTrainingRuns.mockResolvedValue([QUEUED_RUN]);

    renderPage();

    await screen.findByRole('list', { name: 'Model requests' });
    expect(screen.queryByRole('button', { name: /Retry training/i })).toBeNull();
  });

  it('offers no Retry training while a runner still holds a live lease', async () => {
    const claimed: TrainingRun = {
      ...STALE_SUBMITTED_RUN,
      state: 'claimed',
      jobId: undefined,
      claim: {
        owner: 'adam@tillicum',
        claimedAt: new Date(Date.now() - 60_000).toISOString(),
        expiresAt: new Date(Date.now() + 15 * 60_000).toISOString(),
      },
    };
    fetchCourseModelRequest.mockResolvedValue(STALE_REQUEST);
    fetchCourseTrainingRuns.mockResolvedValue([claimed]);

    renderPage();

    await screen.findByRole('list', { name: 'Model requests' });
    expect(screen.queryByRole('button', { name: /Retry training/i })).toBeNull();
  });

  it('offers Retry training once a claim has expired', async () => {
    fetchCourseModelRequest.mockResolvedValue(STALE_REQUEST);
    fetchCourseTrainingRuns.mockResolvedValue([
      {
        ...STALE_SUBMITTED_RUN,
        state: 'claimed' as const,
        jobId: undefined,
        claim: {
          owner: 'adam@tillicum',
          claimedAt: new Date(Date.now() - 2 * 3_600_000).toISOString(),
          expiresAt: new Date(Date.now() - 3_600_000).toISOString(),
        },
      },
    ]);

    renderPage();

    expect(
      await screen.findByRole('button', { name: /Retry training/i }),
    ).toBeInTheDocument();
  });

  it('asks for confirmation before retiring anything', async () => {
    fetchCourseModelRequest.mockResolvedValue(STALE_REQUEST);
    fetchCourseTrainingRuns.mockResolvedValue([STALE_SUBMITTED_RUN]);

    renderPage();
    const dialog = await openRetryConfirmation();

    // The dialog names the run and job being retired, and says the data is kept.
    expect(within(dialog).getByText(new RegExp(STALE_SUBMITTED_RUN.runId))).toBeInTheDocument();
    expect(within(dialog).getByText(/job 253552/)).toBeInTheDocument();
    expect(within(dialog).getByText(/same prepared dataset/)).toBeInTheDocument();
    // Nothing has happened yet.
    expect(retryTrainingForRequest).not.toHaveBeenCalled();
  });

  it('does nothing when the confirmation is cancelled', async () => {
    fetchCourseModelRequest.mockResolvedValue(STALE_REQUEST);
    fetchCourseTrainingRuns.mockResolvedValue([STALE_SUBMITTED_RUN]);

    renderPage();
    const dialog = await openRetryConfirmation();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));

    await waitFor(() => {
      expect(screen.queryByRole('alertdialog')).toBeNull();
    });
    expect(retryTrainingForRequest).not.toHaveBeenCalled();
  });

  it('calls the retry API for that exact course once confirmed', async () => {
    fetchCourseModelRequest.mockResolvedValue(STALE_REQUEST);
    fetchCourseTrainingRuns.mockResolvedValue([STALE_SUBMITTED_RUN]);

    renderPage();
    const dialog = await openRetryConfirmation();
    fireEvent.click(within(dialog).getByRole('button', { name: /Retry training/i }));

    await waitFor(() => {
      expect(retryTrainingForRequest).toHaveBeenCalledWith(COURSE_ID);
    });
    // The browser sends the course and nothing else; the backend decides the rest.
    expect(retryTrainingForRequest).toHaveBeenCalledTimes(1);
  });

  it('reports what was retired and what was queued', async () => {
    fetchCourseModelRequest.mockResolvedValue(STALE_REQUEST);
    fetchCourseTrainingRuns.mockResolvedValue([STALE_SUBMITTED_RUN]);

    renderPage();
    const dialog = await openRetryConfirmation();
    fireEvent.click(within(dialog).getByRole('button', { name: /Retry training/i }));

    expect(
      await screen.findByText(
        new RegExp(
          `Retired ${STALE_SUBMITTED_RUN.runId} and queued full run ${REPLACEMENT_RUN.runId}`,
        ),
      ),
    ).toBeInTheDocument();
  });

  it('re-reads the course after a retry, showing both runs and no stale job', async () => {
    fetchCourseModelRequest.mockResolvedValue(STALE_REQUEST);
    fetchCourseTrainingRuns.mockResolvedValue([STALE_SUBMITTED_RUN]);

    renderPage();
    const dialog = await openRetryConfirmation();

    // What storage returns on the reload the retry triggers.
    fetchCourseModelRequest.mockResolvedValue({
      ...STALE_REQUEST,
      status: 'preparing' as const,
      currentRunId: REPLACEMENT_RUN.runId,
      training: undefined,
    });
    fetchCourseTrainingRuns.mockResolvedValue([SUPERSEDED_RUN, REPLACEMENT_RUN]);

    fireEvent.click(within(dialog).getByRole('button', { name: /Retry training/i }));

    const requests = await screen.findByRole('list', { name: 'Model requests' });

    // The new run is the current one.
    await waitFor(() => {
      expect(within(requests).getByText(REPLACEMENT_RUN.runId)).toBeInTheDocument();
    });
    expect(within(requests).getByText(/run queued/)).toBeInTheDocument();

    // The old run is still there, terminal, with its job id and its reason.
    expect(within(requests).getByText(STALE_SUBMITTED_RUN.runId)).toBeInTheDocument();
    expect(
      within(requests).getByText(/Earlier run:.*failed.*job 253552.*Superseded by admin retry/s),
    ).toBeInTheDocument();

    // The stale active training job is no longer presented as current.
    expect(within(requests).queryByText(/Training job:/)).toBeNull();
  });

  it('surfaces a refused retry to the admin', async () => {
    fetchCourseModelRequest.mockResolvedValue(STALE_REQUEST);
    fetchCourseTrainingRuns.mockResolvedValue([STALE_SUBMITTED_RUN]);
    retryTrainingForRequest.mockRejectedValue(
      new Error('This run is already queued and waiting for a worker.'),
    );

    renderPage();
    const dialog = await openRetryConfirmation();
    fireEvent.click(within(dialog).getByRole('button', { name: /Retry training/i }));

    expect(
      await screen.findByText(/already queued and waiting for a worker/),
    ).toBeInTheDocument();
  });

  it('surfaces an unreachable backend rather than implying a retry happened', async () => {
    fetchCourseModelRequest.mockResolvedValue(STALE_REQUEST);
    fetchCourseTrainingRuns.mockResolvedValue([STALE_SUBMITTED_RUN]);
    retryTrainingForRequest.mockRejectedValue(
      new Error('The service could not be reached.'),
    );

    renderPage();
    const dialog = await openRetryConfirmation();
    fireEvent.click(within(dialog).getByRole('button', { name: /Retry training/i }));

    expect(
      await screen.findByText(/The service could not be reached/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Retired run-/)).toBeNull();
  });

  /* ------------------------------------------------------------------------ *
   * Training a new version of a model that already exists
   *
   * CSS 350 reached `ready` with v1 registered, a succeeded run, and 37/5 still
   * prepared — and the page offered no way to train again. `canQueueTraining`
   * requires `preparing`, so the Queue training control disappears the moment a
   * course succeeds.
   *
   * Retry was not the answer and these assert why: it retires the run a course is
   * waiting on, which would rewrite a succeeded run as failed to get a side
   * effect.
   * ------------------------------------------------------------------------ */

  describe('training a new version', () => {
    async function openRetrainConfirmation() {
      fireEvent.click(await screen.findByRole('button', { name: /Train new version/i }));
      return screen.findByRole('alertdialog');
    }

    it('offers Train new version for a ready course with a finished run', async () => {
      fetchCourseModelRequest.mockResolvedValue(READY_REQUEST);
      fetchCourseTrainingRuns.mockResolvedValue([READY_SUCCEEDED_RUN]);

      renderPage();

      expect(
        await screen.findByRole('button', { name: /Train new version/i }),
      ).toBeInTheDocument();
    });

    it('does not offer Retry training for a run that succeeded', async () => {
      // Retry is disaster recovery. A succeeded run has nothing to recover from,
      // and retiring it would destroy a good result to reuse a code path.
      fetchCourseModelRequest.mockResolvedValue(READY_REQUEST);
      fetchCourseTrainingRuns.mockResolvedValue([READY_SUCCEEDED_RUN]);

      renderPage();
      await screen.findByRole('button', { name: /Train new version/i });

      expect(screen.queryByRole('button', { name: /Retry training/i })).toBeNull();
    });

    it('does not offer it while a run is still outstanding', async () => {
      fetchCourseModelRequest.mockResolvedValue(READY_REQUEST);
      fetchCourseTrainingRuns.mockResolvedValue([
        READY_SUCCEEDED_RUN,
        NEW_VERSION_RUN,
      ]);

      renderPage();
      await screen.findAllByText(new RegExp(NEW_VERSION_RUN.runId));

      expect(screen.queryByRole('button', { name: /Train new version/i })).toBeNull();
    });

    it('does not offer it for a course still being prepared', async () => {
      // That course gets Queue training instead. Two controls for one state
      // would be two ways to do the same thing.
      fetchCourseModelRequest.mockResolvedValue(PREPARED_REQUEST);
      fetchCourseTrainingRuns.mockResolvedValue([]);

      renderPage();
      await screen.findByRole('button', { name: /Queue training/i });

      expect(screen.queryByRole('button', { name: /Train new version/i })).toBeNull();
    });

    it('asks for confirmation and names the dataset it will reuse', async () => {
      fetchCourseModelRequest.mockResolvedValue(READY_REQUEST);
      fetchCourseTrainingRuns.mockResolvedValue([READY_SUCCEEDED_RUN]);

      renderPage();
      const dialog = await openRetrainConfirmation();

      expect(within(dialog).getByText(/37 train \/ 5 validation/)).toBeInTheDocument();
      expect(within(dialog).getByText(/Nothing is re-exported/)).toBeInTheDocument();
      expect(within(dialog).getByText(/GPU time/)).toBeInTheDocument();
      // Nothing has happened yet.
      expect(queueNewVersionForRequest).not.toHaveBeenCalled();
    });

    it('says earlier runs and the current model are kept', async () => {
      fetchCourseModelRequest.mockResolvedValue(READY_REQUEST);
      fetchCourseTrainingRuns.mockResolvedValue([READY_SUCCEEDED_RUN]);

      renderPage();
      const dialog = await openRetrainConfirmation();

      expect(within(dialog).getByText(/Every earlier run is kept/)).toBeInTheDocument();
      expect(
        within(dialog).getByText(/stays registered and keeps serving/),
      ).toBeInTheDocument();
    });

    it('does nothing when the confirmation is cancelled', async () => {
      fetchCourseModelRequest.mockResolvedValue(READY_REQUEST);
      fetchCourseTrainingRuns.mockResolvedValue([READY_SUCCEEDED_RUN]);

      renderPage();
      const dialog = await openRetrainConfirmation();
      fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));

      await waitFor(() => {
        expect(screen.queryByRole('alertdialog')).toBeNull();
      });
      expect(queueNewVersionForRequest).not.toHaveBeenCalled();
    });

    it('queues the run once confirmed, passing the request and its history', async () => {
      fetchCourseModelRequest.mockResolvedValue(READY_REQUEST);
      fetchCourseTrainingRuns.mockResolvedValue([READY_SUCCEEDED_RUN]);

      renderPage();
      const dialog = await openRetrainConfirmation();
      fireEvent.click(within(dialog).getByRole('button', { name: /Queue new run/i }));

      await waitFor(() => {
        expect(queueNewVersionForRequest).toHaveBeenCalledWith(
          COURSE_ID,
          READY_REQUEST,
          [READY_SUCCEEDED_RUN],
        );
      });
      expect(queueNewVersionForRequest).toHaveBeenCalledTimes(1);
      expect(retryTrainingForRequest).not.toHaveBeenCalled();
    });

    it('reports the new run and that nothing was replaced', async () => {
      fetchCourseModelRequest.mockResolvedValue(READY_REQUEST);
      fetchCourseTrainingRuns.mockResolvedValue([READY_SUCCEEDED_RUN]);

      renderPage();
      const dialog = await openRetrainConfirmation();
      fireEvent.click(within(dialog).getByRole('button', { name: /Queue new run/i }));

      expect(
        await screen.findAllByText(new RegExp(NEW_VERSION_RUN.runId)),
      ).not.toHaveLength(0);
      expect(
        await screen.findByText(/Earlier runs and the current model version are unchanged/),
      ).toBeInTheDocument();
    });

    it('surfaces a duplicate refusal without claiming anything was queued', async () => {
      fetchCourseModelRequest.mockResolvedValue(READY_REQUEST);
      fetchCourseTrainingRuns.mockResolvedValue([READY_SUCCEEDED_RUN]);
      queueNewVersionForRequest.mockRejectedValue(new DuplicateTrainingRunError());

      renderPage();
      const dialog = await openRetrainConfirmation();
      fireEvent.click(within(dialog).getByRole('button', { name: /Queue new run/i }));

      expect(
        await screen.findByText(/already queued or under way/i),
      ).toBeInTheDocument();
    });

    it('does not offer it when there is no dataset to reuse', async () => {
      const { preparation, ...withoutPreparation } = READY_REQUEST;
      void preparation;
      fetchCourseModelRequest.mockResolvedValue(withoutPreparation);
      fetchCourseTrainingRuns.mockResolvedValue([
        { ...READY_SUCCEEDED_RUN, datasetRef: '' },
      ]);

      renderPage();
      await screen.findAllByText(new RegExp(READY_SUCCEEDED_RUN.runId));

      expect(screen.queryByRole('button', { name: /Train new version/i })).toBeNull();
    });
  });
});
