// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

/**
 * The panel that used to say "No job tracking yet".
 *
 * That was an honest statement of a real gap: a finished job reported nothing
 * back, so the application genuinely could not say more than "a run was queued".
 * It is also how Slurm job 253552 sat at `submitted` for days with a trained
 * adapter already on the cluster and no screen anywhere able to show it.
 *
 * These cover what the panel now has to get right: the states an operator asks
 * about, the cluster detail that belongs on an admin surface, and the
 * distinction between a model being ready and something actually serving it.
 */

const subscribeToCoursesMock = vi.fn();
const fetchCourseTrainingRuns = vi.fn();
const fetchCourseModel = vi.fn();
const fetchCurrentServingSession = vi.fn();

vi.mock('../../lib/coursesDb', () => ({
  subscribeToCourses: (...args: unknown[]) => subscribeToCoursesMock(...args),
}));

vi.mock('../../lib/trainingRunDb', async () => {
  const actual = await vi.importActual<typeof import('../../lib/trainingRunDb')>(
    '../../lib/trainingRunDb',
  );
  return {
    ...actual,
    fetchCourseTrainingRuns: (...args: unknown[]) => fetchCourseTrainingRuns(...args),
  };
});

vi.mock('../../lib/courseModelDb', async () => {
  const actual = await vi.importActual<typeof import('../../lib/courseModelDb')>(
    '../../lib/courseModelDb',
  );
  return {
    ...actual,
    fetchCourseModel: (...args: unknown[]) => fetchCourseModel(...args),
  };
});

vi.mock('../../lib/servingSessionDb', () => ({
  fetchCurrentServingSession: () => fetchCurrentServingSession(),
}));

const { TrainingJobs } = await import('./TrainingJobs');

const CSS350 = 'css-350-spring-2026-n3h9';
const CSS360 = 'css-360-winter-2026-a7rp';
const RUN_ID = 'run-20260827t064701z-1cf650';

const SUCCEEDED_RUN = {
  runId: RUN_ID,
  courseId: CSS350,
  mode: 'full' as const,
  state: 'succeeded' as const,
  enqueuedAt: '2026-08-27T06:40:00.000Z',
  updatedAt: '2026-08-27T06:48:00.000Z',
  datasetRef: `exports/${CSS350}`,
  approvedExampleCount: 42,
  trainExamples: 37,
  validationExamples: 5,
  attempt: 1,
  jobId: '264787',
  completion: {
    outcome: 'succeeded' as const,
    receivedAt: '2026-08-27T06:48:00.000Z',
    jobId: '264787',
    intendedOptimizerSteps: 15,
    completedSteps: 15,
    missingOptimizerSteps: 0,
    trainingLengthSatisfied: true,
    trainLoss: 1.2345,
    evalLoss: 1.4567,
    actualGpuHours: 0.0134,
    gpuCount: 1,
    elapsedSeconds: 48.2,
    gitCommitSha: '9941833cafe0000000000000000000000000beef',
    datasetVersion: `${CSS350}-approved-split-seed360-n42`,
    outputRef: `qlora-runs/${CSS350}/20260827T064701Z-full`,
    artifactRef: `qlora-runs/${CSS350}/20260827T064701Z-full/adapter`,
  },
};

const READY_VERSION = {
  version: 'v1',
  baseModel: 'meta-llama/Llama-3.2-3B-Instruct',
  trainingExampleCount: 37,
  status: 'ready' as const,
  deployment: 'offline' as const,
  artifactRef: `qlora-runs/${CSS350}/20260827T064701Z-full/adapter`,
  createdAt: '2026-08-27T06:48:00.000Z',
  runId: RUN_ID,
};

function courses(ids: string[]) {
  return ids.map((courseId) => ({
    courseId,
    metadata: {
      name: courseId.startsWith('css-350') ? 'CSS 350' : 'CSS 360',
      title: 'Software Engineering',
    },
  }));
}

function mockCourses(ids: string[]) {
  subscribeToCoursesMock.mockImplementation((onData: (value: unknown[]) => void) => {
    onData(courses(ids));
    return () => {};
  });
}

// Vitest globals are off in this project, so Testing Library's automatic
// cleanup never registers. Without this every render stacks into the same
// document and `findByRole` sees several copies of the same list.
afterEach(cleanup);

beforeEach(() => {
  subscribeToCoursesMock.mockReset();
  fetchCourseTrainingRuns.mockReset();
  fetchCourseModel.mockReset();
  fetchCurrentServingSession.mockReset();

  mockCourses([CSS350]);
  fetchCourseTrainingRuns.mockResolvedValue([]);
  fetchCourseModel.mockResolvedValue(null);
  fetchCurrentServingSession.mockResolvedValue(null);
});

describe('training jobs', () => {
  it('shows a run with its Slurm job, counts and state', async () => {
    fetchCourseTrainingRuns.mockResolvedValue([SUCCEEDED_RUN]);

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent(RUN_ID);
    expect(list).toHaveTextContent('264787');
    expect(list).toHaveTextContent('37 train / 5 validation');
    expect(list).toHaveTextContent('from 42 approved');
    expect(list).toHaveTextContent('succeeded');
  });

  it('shows the optimizer-step accounting a full run reported', async () => {
    /*
     * The number the QLoRA truncation fix exists to make visible: a run that
     * completed 12 of 15 steps looked identical to a healthy one from every
     * screen before this.
     */
    fetchCourseTrainingRuns.mockResolvedValue([SUCCEEDED_RUN]);

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent('15/15');
  });

  it('says plainly when a run finished short of its step budget', async () => {
    fetchCourseTrainingRuns.mockResolvedValue([
      {
        ...SUCCEEDED_RUN,
        completion: {
          ...SUCCEEDED_RUN.completion,
          completedSteps: 12,
          missingOptimizerSteps: 3,
          trainingLengthSatisfied: false,
        },
      },
    ]);

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent('12/15');
    expect(list).toHaveTextContent('finished short of its budget');
  });

  it('shows measured GPU cost rather than what was requested', async () => {
    fetchCourseTrainingRuns.mockResolvedValue([SUCCEEDED_RUN]);

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent('0.0134 GPU-h');
    expect(list).toHaveTextContent('48.2s elapsed');
  });

  it('shows the git commit and dataset version a model was built from', async () => {
    fetchCourseTrainingRuns.mockResolvedValue([SUCCEEDED_RUN]);

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent('9941833cafe0');
    expect(list).toHaveTextContent(`${CSS350}-approved-split-seed360-n42`);
  });

  it('shows the model version a successful run registered', async () => {
    fetchCourseTrainingRuns.mockResolvedValue([SUCCEEDED_RUN]);
    fetchCourseModel.mockResolvedValue({
      currentVersion: 'v1',
      versions: { v1: READY_VERSION },
    });

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent('v1 · ready · not published');
  });

  it('labels a retried run as superseded rather than as a plain failure', async () => {
    /*
     * A retry stores the retired run as `failed` with a specific reason, because
     * adding a state would make older browser bundles drop it from the history
     * the feature exists to preserve. The label is reconstructed here, where it
     * is only a label.
     */
    fetchCourseTrainingRuns.mockResolvedValue([
      {
        ...SUCCEEDED_RUN,
        state: 'failed' as const,
        error: 'Superseded by admin retry',
        completion: undefined,
      },
    ]);

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent('superseded');
  });

  it('shows a failed run with the stage and the operator message', async () => {
    fetchCourseTrainingRuns.mockResolvedValue([
      {
        ...SUCCEEDED_RUN,
        state: 'failed' as const,
        error: 'CUDA out of memory during step 4.',
        completion: {
          outcome: 'failed' as const,
          receivedAt: '2026-08-27T06:48:00.000Z',
          failureStage: 'training',
          error: 'CUDA out of memory during step 4.',
        },
      },
    ]);

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent('failed');
    expect(list).toHaveTextContent('Failed at');
    expect(list).toHaveTextContent('CUDA out of memory');
  });

  it('shows a queued run that has no job and no report yet', async () => {
    fetchCourseTrainingRuns.mockResolvedValue([
      {
        ...SUCCEEDED_RUN,
        state: 'queued' as const,
        jobId: undefined,
        attempt: 0,
        completion: undefined,
      },
    ]);

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent('queued');
    expect(list).toHaveTextContent('not submitted');
  });

  it('shows who is holding a claimed run and until when', async () => {
    fetchCourseTrainingRuns.mockResolvedValue([
      {
        ...SUCCEEDED_RUN,
        state: 'claimed' as const,
        jobId: undefined,
        completion: undefined,
        claim: {
          owner: 'testuser@tillicum',
          claimedAt: '2026-08-27T06:41:00.000Z',
          expiresAt: '2026-08-27T06:56:00.000Z',
        },
      },
    ]);

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent('testuser@tillicum');
  });

  it('keeps each course to its own runs', async () => {
    mockCourses([CSS350, CSS360]);
    fetchCourseTrainingRuns.mockImplementation(async (courseId: string) =>
      courseId === CSS350
        ? [SUCCEEDED_RUN]
        : [{ ...SUCCEEDED_RUN, runId: 'run-css360', courseId: CSS360 }],
    );

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent(CSS350);
    expect(list).toHaveTextContent(CSS360);
    expect(fetchCourseTrainingRuns).toHaveBeenCalledWith(CSS350);
    expect(fetchCourseTrainingRuns).toHaveBeenCalledWith(CSS360);
  });

  it('orders runs newest first across every course', async () => {
    mockCourses([CSS350, CSS360]);
    fetchCourseTrainingRuns.mockImplementation(async (courseId: string) =>
      courseId === CSS350
        ? [{ ...SUCCEEDED_RUN, enqueuedAt: '2026-08-01T00:00:00.000Z' }]
        : [
            {
              ...SUCCEEDED_RUN,
              runId: 'run-newer',
              courseId: CSS360,
              enqueuedAt: '2026-09-01T00:00:00.000Z',
            },
          ],
    );

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    const items = list.textContent ?? '';
    expect(items.indexOf('run-newer')).toBeLessThan(items.indexOf(RUN_ID));
  });

  it('shows an empty state when nothing has ever been queued', async () => {
    render(<TrainingJobs />);

    expect(await screen.findByText('No training runs yet')).toBeInTheDocument();
  });

  it('still lists runs when a course registry cannot be read', async () => {
    // One unreachable record must not blank the whole table.
    fetchCourseTrainingRuns.mockResolvedValue([SUCCEEDED_RUN]);
    fetchCourseModel.mockRejectedValue(new Error('registry unavailable'));

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent(RUN_ID);
  });
});

describe('serving session banner', () => {
  it('says nothing is serving when no session is recorded', async () => {
    /*
     * The resting state, and the one that keeps `ready` and `deployed` distinct:
     * a trained model exists whether or not a GPU is running.
     */
    render(<TrainingJobs />);

    expect(
      await screen.findByText('No fine-tuned service is running'),
    ).toBeInTheDocument();
  });

  it('shows a live session with its expiry and what it serves', async () => {
    fetchCurrentServingSession.mockResolvedValue({
      sessionId: 'serve-264790',
      jobId: '264790',
      state: 'ready',
      startedAt: '2026-08-27T12:00:00.000Z',
      expiresAt: '2026-08-27T14:00:00.000Z',
      live: true,
      courses: [{ courseId: CSS350, currentVersion: 'v1' }],
    });

    render(<TrainingJobs />);

    expect(
      await screen.findByText('A fine-tuned service is running'),
    ).toBeInTheDocument();
    expect(screen.getByText(/serve-264790/)).toBeInTheDocument();
    expect(screen.getByText(new RegExp(`${CSS350} \\(v1\\)`))).toBeInTheDocument();
  });

  it('treats an expired session as nothing serving', async () => {
    fetchCurrentServingSession.mockResolvedValue({
      sessionId: 'serve-264790',
      jobId: '264790',
      state: 'expired',
      live: false,
    });

    render(<TrainingJobs />);

    expect(
      await screen.findByText('No fine-tuned service is running'),
    ).toBeInTheDocument();
  });

  it('reports a failure to read the session without hiding the runs', async () => {
    fetchCourseTrainingRuns.mockResolvedValue([SUCCEEDED_RUN]);
    fetchCurrentServingSession.mockRejectedValue(new Error('backend unreachable'));

    render(<TrainingJobs />);

    expect(
      await screen.findByText('Could not read the serving session'),
    ).toBeInTheDocument();
    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent(RUN_ID);
  });
});

/* ------------------------------------------------------------------------ *
 * Published is not the same as serving, and an infrastructure failure is not
 * the same as a training failure
 *
 * Both distinctions came out of the real end-to-end run. `deployment = online`
 * means the adapter is in the cluster's serving tree — durable — while whether
 * a GPU session is up right now is the serving-session banner's business. And a
 * run that died in preflight (`Failed to get device handle for GPU 0`) needs a
 * different response from an admin than one whose training went wrong.
 * ------------------------------------------------------------------------ */

describe('published versus serving', () => {
  it('calls a published version published, not in use', async () => {
    fetchCourseTrainingRuns.mockResolvedValue([SUCCEEDED_RUN]);
    fetchCourseModel.mockResolvedValue({
      currentVersion: 'v1',
      versions: { v1: { ...READY_VERSION, deployment: 'online' } },
    });

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent('v1 · ready · published');
  });

  it('says a ready version is not published rather than offline', async () => {
    // "offline" read as "the service is down". It is not — the version simply
    // has not been copied to the cluster yet.
    fetchCourseTrainingRuns.mockResolvedValue([SUCCEEDED_RUN]);
    fetchCourseModel.mockResolvedValue({
      currentVersion: 'v1',
      versions: { v1: { ...READY_VERSION, deployment: 'offline' } },
    });

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent('v1 · ready · not published');
  });

  it('does not claim a published version is being served right now', async () => {
    fetchCourseTrainingRuns.mockResolvedValue([SUCCEEDED_RUN]);
    fetchCourseModel.mockResolvedValue({
      currentVersion: 'v1',
      versions: { v1: { ...READY_VERSION, deployment: 'online' } },
    });
    fetchCurrentServingSession.mockResolvedValue(null);

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).not.toHaveTextContent('in use');
    expect(list).not.toHaveTextContent('serving now');
  });

  it('reports an unrecorded deployment as unknown rather than guessing', async () => {
    fetchCourseTrainingRuns.mockResolvedValue([SUCCEEDED_RUN]);
    fetchCourseModel.mockResolvedValue({
      currentVersion: 'v1',
      versions: { v1: { ...READY_VERSION, deployment: 'unknown' } },
    });

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent('publication unknown');
  });
});

describe('infrastructure failures', () => {
  function failedRun(stage: string, error: string) {
    return {
      ...SUCCEEDED_RUN,
      state: 'failed' as const,
      error,
      completion: {
        outcome: 'failed' as const,
        receivedAt: '2026-09-02T08:10:00.000Z',
        jobId: '265300',
        failureStage: stage,
        error,
      },
    };
  }

  it('says plainly that a preflight failure trained nothing', async () => {
    /*
     * The real one: a node whose GPU had gone. The run never trained, so the
     * dataset and every earlier version are untouched and a retry is the whole
     * remedy — which is not obvious from "exited with status 6".
     */
    fetchCourseTrainingRuns.mockResolvedValue([
      failedRun('preflight', 'The Slurm job exited with status 6.'),
    ]);

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent('preflight failed before training started');
    expect(list).toHaveTextContent('Retry training');
  });

  it('keeps the operator-facing error alongside the plain summary', async () => {
    // The summary is for deciding what to do; the raw text is for debugging.
    fetchCourseTrainingRuns.mockResolvedValue([
      failedRun('preflight', 'The Slurm job exited with status 6.'),
    ]);

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent('exited with status 6');
    expect(list).toHaveTextContent('preflight');
  });

  it('does not blame the hardware it cannot diagnose', async () => {
    fetchCourseTrainingRuns.mockResolvedValue([
      failedRun('preflight', 'The Slurm job exited with status 6.'),
    ]);

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    // No node name, no claim about which GPU or why.
    expect(list).not.toHaveTextContent('g018');
    expect(list).not.toHaveTextContent('broken');
  });

  it('describes a training failure differently from an infrastructure one', async () => {
    fetchCourseTrainingRuns.mockResolvedValue([
      failedRun('training', 'Loss became NaN at step 4.'),
    ]);

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent('Check the run logs before retrying');
    expect(list).not.toHaveTextContent('preflight failed before training started');
  });

  it('adds no summary for a failure stage it does not recognise', async () => {
    // Better to show only the real error than to invent an explanation.
    fetchCourseTrainingRuns.mockResolvedValue([
      failedRun('something-new', 'Unrecognised failure.'),
    ]);

    render(<TrainingJobs />);

    const list = await screen.findByRole('list', { name: 'Training jobs' });
    expect(list).toHaveTextContent('Unrecognised failure.');
    expect(list).not.toHaveTextContent('Retry training to queue a replacement');
  });
});
