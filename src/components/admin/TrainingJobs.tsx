import { useEffect, useState } from 'react';
import { Callout } from '../ui/Callout';
import { EmptyState } from '../ui/EmptyState';
import { SectionHeader } from '../ui/SectionHeader';
import { StatusPill } from '../ui/StatusPill';
import { formatCourseHeading } from '../../lib/courseLabels';
import { fetchCourseModel } from '../../lib/courseModelDb';
import { fetchCourseTrainingRuns } from '../../lib/trainingRunDb';
import { fetchCurrentServingSession } from '../../lib/servingSessionDb';
import { useCourses } from '../../hooks/useCourses';
import type {
  CourseModelVersion,
  ServingSession,
  TrainingRun,
} from '../../types';

/**
 * What every training run across every course is actually doing.
 *
 * This replaces a panel that said "No job tracking yet". That was accurate when
 * a finished job reported nothing back — the application genuinely could not
 * say more than "a run was queued" — and it was how a real successful run
 * (Slurm job 253552) sat at `submitted` for days with nobody able to see, from
 * any screen, that a trained adapter was already sitting on the cluster.
 *
 * Everything here comes from PostgreSQL. The browser never queries Slurm: it
 * reads what the cluster reported through the queue API, which is the only
 * thing that survives a browser being closed. A run with no `completion` and a
 * terminal state is therefore a run that predates completion reporting, not a
 * gap in this component.
 *
 * Admin surface, so cluster detail belongs here: job ids, attempts, optimizer
 * steps, GPU hours, git commit. The professor's own page keeps the five plain
 * states and shows none of it.
 */

interface CourseRuns {
  courseId: string;
  name: string;
  runs: TrainingRun[];
  versionForRun: Record<string, CourseModelVersion>;
}

interface JobRow {
  courseId: string;
  courseName: string;
  run: TrainingRun;
  version: CourseModelVersion | null;
}

function runTone(state: TrainingRun['state']) {
  switch (state) {
    case 'queued':
      return 'info' as const;
    case 'claimed':
    case 'submitted':
    case 'training':
      return 'progress' as const;
    case 'succeeded':
      return 'success' as const;
    case 'failed':
      return 'danger' as const;
    default:
      return 'neutral' as const;
  }
}

/**
 * A retried run is stored as `failed` with a specific reason rather than as a
 * state of its own — adding a state would make every browser running an older
 * bundle drop the run from the history the feature exists to preserve. The
 * label is reconstructed from that reason here, where it is only a label.
 */
const SUPERSEDED_ERROR = 'Superseded by admin retry';

function isSuperseded(run: TrainingRun): boolean {
  return run.state === 'failed' && (run.error ?? '').startsWith(SUPERSEDED_ERROR);
}

/**
 * What a version's `deployment` means, in words.
 *
 * `online` records that the adapter has been published into the cluster's
 * serving tree and is the version inference resolves — a durable fact. It does
 * not record that a GPU session is up this second; that is what the serving
 * session banner above is for, and conflating the two is how "in use" ended up
 * on a page describing a course nothing was serving.
 */
function describeDeployment(deployment: string): string {
  switch (deployment) {
    case 'online':
      return 'published';
    case 'offline':
      return 'not published';
    default:
      return 'publication unknown';
  }
}

/**
 * A short, honest headline for a failed run.
 *
 * The point is to separate "the cluster was broken" from "the training was
 * wrong", because they need completely different responses from an admin and
 * the raw text does not distinguish them. A preflight failure — the real one
 * here was `Failed to get device handle for GPU 0` on a node whose GPU had gone
 * — is retryable as-is; a data or configuration failure is not.
 *
 * Deliberately does not diagnose hardware. It says which stage failed and what
 * an admin can do, and leaves the operator-facing error text below it.
 */
function describeFailure(
  stage: string | undefined,
  error: string | undefined,
): string | null {
  if (!stage && !error) {
    return null;
  }
  switch (stage) {
    case 'preflight':
      return 'Node or GPU preflight failed before training started — nothing was trained. Retry training to queue a replacement run.';
    case 'submission':
      return 'The job was never submitted to Slurm. The dataset is unchanged; queue it again.';
    case 'dataset':
      return 'The prepared dataset did not validate on the cluster. Rebuild the dataset and prepare the split again.';
    case 'training':
      return 'Training started and did not finish. Check the run logs before retrying.';
    default:
      return null;
  }
}

function stateLabel(run: TrainingRun): string {
  return isSuperseded(run) ? 'superseded' : run.state;
}

function formatTime(value: string | undefined): string {
  if (!value) {
    return '—';
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function formatDuration(seconds: number | undefined): string {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds < 0) {
    return '—';
  }
  if (seconds < 90) {
    return `${seconds.toFixed(1)}s`;
  }
  if (seconds < 5400) {
    return `${(seconds / 60).toFixed(1)}m`;
  }
  return `${(seconds / 3600).toFixed(2)}h`;
}

function formatGpuHours(hours: number | undefined): string {
  if (typeof hours !== 'number' || !Number.isFinite(hours)) {
    return '—';
  }
  return `${hours.toFixed(4)} GPU-h`;
}

function formatLoss(value: number | undefined): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toFixed(4)
    : '—';
}

/**
 * Newest first. Runs are stored oldest-first per course because that is the
 * order a queue is consumed in; an operator asking "what is happening?" is
 * asking about the most recent thing.
 */
function flatten(courses: CourseRuns[]): JobRow[] {
  const rows: JobRow[] = [];
  for (const course of courses) {
    for (const run of course.runs) {
      rows.push({
        courseId: course.courseId,
        courseName: course.name,
        run,
        version: course.versionForRun[run.runId] ?? null,
      });
    }
  }
  return rows.sort((left, right) =>
    right.run.enqueuedAt.localeCompare(left.run.enqueuedAt),
  );
}

function ServingSessionBanner({ session }: { session: ServingSession | null }) {
  if (!session || !session.live) {
    return (
      <Callout tone="info" title="No fine-tuned service is running">
        Ready models exist independently of whether anything is serving them.
        Start a session on Tillicum with{' '}
        <code>./training/start_finetuned_service.sh</code>, then open the tunnel
        on the VM with <code>./scripts/start_finetuned_tunnel.sh --from-backend</code>.
      </Callout>
    );
  }

  const courses = session.courses ?? [];
  return (
    <Callout tone="success" title="A fine-tuned service is running">
      <p>
        Session <code>{session.sessionId}</code> (Slurm job{' '}
        <code>{session.jobId}</code>) started {formatTime(session.startedAt)} and
        ends {formatTime(session.expiresAt)}.
      </p>
      <p>
        {courses.length > 0
          ? `Serving: ${courses
              .map((course) => `${course.courseId} (${course.currentVersion})`)
              .join(', ')}.`
          : 'No course adapters are published to this session yet.'}
      </p>
    </Callout>
  );
}

function JobDetails({ row }: { row: JobRow }) {
  const { run, version } = row;
  const completion = run.completion;

  return (
    <dl className="admin-example__meta">
      <div>
        <dt>Run</dt>
        <dd>
          <code>{run.runId}</code>
        </dd>
      </div>
      <div>
        <dt>Slurm job</dt>
        <dd>{run.jobId ? <code>{run.jobId}</code> : 'not submitted'}</dd>
      </div>
      <div>
        <dt>Mode</dt>
        <dd>{run.mode}</dd>
      </div>
      <div>
        <dt>Attempt</dt>
        <dd>{run.attempt}</dd>
      </div>
      <div>
        <dt>Examples</dt>
        <dd>
          {run.trainExamples} train / {run.validationExamples} validation
          {run.approvedExampleCount > 0
            ? ` (from ${run.approvedExampleCount} approved)`
            : ''}
        </dd>
      </div>
      <div>
        <dt>Queued</dt>
        <dd>{formatTime(run.enqueuedAt)}</dd>
      </div>
      <div>
        <dt>Last update</dt>
        <dd>{formatTime(run.updatedAt)}</dd>
      </div>
      {run.claim && (
        <div>
          <dt>Held by</dt>
          <dd>
            {run.claim.owner} until {formatTime(run.claim.expiresAt)}
          </dd>
        </div>
      )}
      {completion && (
        <>
          <div>
            <dt>Reported</dt>
            <dd>
              {completion.outcome} at {formatTime(completion.receivedAt)}
            </dd>
          </div>
          <div>
            <dt>Optimizer steps</dt>
            <dd>
              {typeof completion.completedSteps === 'number'
                ? `${completion.completedSteps}/${
                    completion.intendedOptimizerSteps ?? '?'
                  }`
                : '—'}
              {completion.trainingLengthSatisfied === false
                ? ' — finished short of its budget'
                : ''}
            </dd>
          </div>
          <div>
            <dt>Loss</dt>
            <dd>
              train {formatLoss(completion.trainLoss)} / eval{' '}
              {formatLoss(completion.evalLoss)}
            </dd>
          </div>
          <div>
            <dt>GPU cost</dt>
            <dd>
              {formatGpuHours(completion.actualGpuHours)}
              {typeof completion.gpuCount === 'number'
                ? ` · ${completion.gpuCount} GPU`
                : ''}
              {' · '}
              {formatDuration(completion.elapsedSeconds)} elapsed
            </dd>
          </div>
          {completion.gitCommitSha && (
            <div>
              <dt>Code</dt>
              <dd>
                <code>{completion.gitCommitSha.slice(0, 12)}</code>
              </dd>
            </div>
          )}
          {completion.datasetVersion && (
            <div>
              <dt>Dataset</dt>
              <dd>
                <code>{completion.datasetVersion}</code>
              </dd>
            </div>
          )}
          {completion.outputRef && (
            <div>
              <dt>Artifact</dt>
              <dd>
                <code>{completion.artifactRef ?? completion.outputRef}</code>
              </dd>
            </div>
          )}
          {completion.failureStage && (
            <div>
              <dt>Failed at</dt>
              <dd>{completion.failureStage}</dd>
            </div>
          )}
        </>
      )}
      {version && (
        <div>
          <dt>Model version</dt>
          <dd>
            {version.version} · {version.status} · {describeDeployment(version.deployment)}
          </dd>
        </div>
      )}
    </dl>
  );
}

export function TrainingJobs() {
  const { state: courses } = useCourses();
  const [rows, setRows] = useState<JobRow[] | null>(null);
  const [session, setSession] = useState<ServingSession | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (courses.status !== 'ready') {
      return;
    }

    let cancelled = false;

    async function load(courseId: string, name: string): Promise<CourseRuns> {
      // Runs and the registry are read separately and each failure is absorbed:
      // a course whose registry cannot be read still has runs worth showing,
      // and one unreachable course must not blank the whole table.
      const runs = await fetchCourseTrainingRuns(courseId).catch(() => []);
      const registry = await fetchCourseModel(courseId).catch(() => null);

      const versionForRun: Record<string, CourseModelVersion> = {};
      for (const version of Object.values(registry?.versions ?? {})) {
        if (version.runId) {
          versionForRun[version.runId] = version;
        }
      }

      return { courseId, name, runs, versionForRun };
    }

    void Promise.all(
      courses.courses.map(({ courseId, metadata }) =>
        load(courseId, formatCourseHeading(metadata.name, metadata.title)),
      ),
    ).then((loaded) => {
      if (!cancelled) {
        setRows(flatten(loaded));
      }
    });

    void fetchCurrentServingSession()
      .then((current) => {
        if (!cancelled) {
          setSession(current);
        }
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setError(
            reason instanceof Error
              ? reason.message
              : 'Could not read the serving session.',
          );
        }
      });

    return () => {
      cancelled = true;
    };
  }, [courses]);

  return (
    <section className="ui-stack ui-stack--snug">
      <SectionHeader
        title="Training jobs"
        description="Every run this application knows about, newest first. Reported by the cluster; the browser never queries Slurm."
        divider
      />

      {error ? (
        <Callout tone="warning" title="Could not read the serving session">
          {error}
        </Callout>
      ) : (
        <ServingSessionBanner session={session} />
      )}

      {rows === null ? (
        <p className="ui-text-muted" role="status" aria-live="polite">
          Reading training runs…
        </p>
      ) : rows.length === 0 ? (
        <EmptyState
          title="No training runs yet"
          description="Queue one from a prepared model request above. It is picked up the next time the queue is run on the cluster."
        />
      ) : (
        <ul className="admin-rows" aria-label="Training jobs">
          {rows.map((row) => (
            <li
              key={`${row.courseId}:${row.run.runId}`}
              className="admin-row admin-row--stacked"
            >
              <div className="admin-row__main">
                <p className="admin-row__label">{row.courseName}</p>
                <p className="admin-row__value">
                  <code>{row.courseId}</code>
                </p>
                <JobDetails row={row} />
                {row.run.error && (
                  <p className="admin-row__error">{row.run.error}</p>
                )}
                {describeFailure(
                  row.run.completion?.failureStage,
                  row.run.completion?.error,
                ) && (
                  <p className="admin-row__value">
                    {describeFailure(
                      row.run.completion?.failureStage,
                      row.run.completion?.error,
                    )}
                  </p>
                )}
                {row.run.completion?.error && (
                  <p className="admin-row__error">
                    {row.run.completion.error}
                  </p>
                )}
              </div>
              <div className="admin-row__actions">
                <StatusPill tone={isSuperseded(row.run) ? 'neutral' : runTone(row.run.state)}>
                  {stateLabel(row.run)}
                </StatusPill>
                {row.version && (
                  <StatusPill
                    tone={
                      row.version.deployment === 'online'
                        ? 'success'
                        : row.version.status === 'ready'
                          ? 'info'
                          : 'neutral'
                    }
                  >
                    {row.version.version} · {describeDeployment(row.version.deployment)}
                  </StatusPill>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
