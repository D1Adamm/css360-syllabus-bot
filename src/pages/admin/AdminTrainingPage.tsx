import { useCallback, useEffect, useState } from 'react';
import { Button } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { EmptyState } from '../../components/ui/EmptyState';
import { formatCourseHeading } from '../../lib/courseLabels';
import { PageHeader } from '../../components/ui/PageHeader';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { useCourseExampleCounts } from '../../hooks/useCourseExampleCounts';
import { useCourses } from '../../hooks/useCourses';
import {
  ApiError,
  exportApprovedCourseSeeds,
  getApprovedExportStatus,
  prepareTrainingSplit,
} from '../../lib/api';

/**
 * Dataset export and train/validation split.
 *
 * These two actions used to sit on the professor review page, where they
 * exposed dataset mechanics and absolute server paths to someone reviewing
 * course content. They are unchanged here — same endpoints, same request
 * bodies, same gating on an existing approved export — just relocated to the
 * audience they were always meant for.
 */

interface CourseTrainingState {
  hasExport: boolean;
  exampleCount: number;
  busy: 'export' | 'split' | null;
  message: string | null;
  error: boolean;
}

const INITIAL: CourseTrainingState = {
  hasExport: false,
  exampleCount: 0,
  busy: null,
  message: null,
  error: false,
};

function CourseTrainingRow({ courseId, name }: { courseId: string; name: string }) {
  const [state, setState] = useState<CourseTrainingState>(INITIAL);
  // Approved counts come from the examples' review status, never from whether
  // an export file happens to exist — an export can be stale, missing, or made
  // before the last few approvals.
  const countsState = useCourseExampleCounts(courseId);
  const counts = countsState.status === 'ready' ? countsState.counts : null;

  const refreshStatus = useCallback(async () => {
    try {
      const status = await getApprovedExportStatus(courseId);
      setState((current) => ({
        ...current,
        hasExport: Boolean(status.exists),
        exampleCount: Number(status.exampleCount) || 0,
      }));
    } catch {
      setState((current) => ({ ...current, hasExport: false, exampleCount: 0 }));
    }
  }, [courseId]);

  useEffect(() => {
    void refreshStatus();
  }, [refreshStatus]);

  async function handleExport() {
    setState((current) => ({ ...current, busy: 'export', message: null, error: false }));
    try {
      const response = await exportApprovedCourseSeeds(courseId);
      const count =
        Number(response.summary?.validatedCount) ||
        Number(response.summary?.exportedCount) ||
        Number(response.summary?.approvedCount) ||
        0;
      const path = String(
        response.summary?.exportPath || response.summary?.files?.finetuneJsonl || '',
      ).trim();
      setState((current) => ({
        ...current,
        busy: null,
        hasExport: true,
        message: `Training dataset created from ${count} approved example${
          count === 1 ? '' : 's'
        }${path ? ` → ${path}` : ''}`,
        error: false,
      }));
      await refreshStatus();
    } catch (error) {
      setState((current) => ({
        ...current,
        busy: null,
        message: error instanceof ApiError ? error.message : 'Export failed.',
        error: true,
      }));
    }
  }

  async function handleSplit() {
    setState((current) => ({ ...current, busy: 'split', message: null, error: false }));
    try {
      const response = await prepareTrainingSplit(courseId);
      const train = Number(response.summary?.trainExamples) || 0;
      const validation = Number(response.summary?.validationExamples) || 0;
      setState((current) => ({
        ...current,
        busy: null,
        message: `Prepared split: ${train} train, ${validation} validation`,
        error: false,
      }));
    } catch (error) {
      setState((current) => ({
        ...current,
        busy: null,
        message:
          error instanceof ApiError ? error.message : 'Prepare training split failed.',
        error: true,
      }));
    }
  }

  const busy = state.busy !== null;

  return (
    <li className="admin-row admin-row--stacked">
      <div className="admin-row__main">
        <p className="admin-row__label">{name}</p>
        <p className="admin-row__value">
          <code>{courseId}</code>
        </p>
        <p className="ui-text-xs ui-text-muted">
          {counts
            ? `${counts.approved} approved example${counts.approved === 1 ? '' : 's'}`
            : countsState.status === 'loading'
              ? 'Counting approved examples…'
              : 'Approved count unavailable'}
        </p>
        <p className="ui-text-xs ui-text-muted">
          Training dataset:{' '}
          {state.hasExport ? (
            <strong>{state.exampleCount} examples</strong>
          ) : (
            'Not created'
          )}
        </p>
        {state.message && (
          <p
            className={`ui-text-xs ${state.error ? 'admin-row__error' : 'ui-text-muted'}`}
            role={state.error ? 'alert' : 'status'}
          >
            {state.message}
          </p>
        )}
      </div>

      <div className="admin-row__actions">
        <Button
          size="sm"
          variant="secondary"
          onClick={() => void handleExport()}
          loading={state.busy === 'export'}
          loadingLabel="Building…"
          disabled={busy}
        >
          {state.hasExport ? 'Rebuild dataset' : 'Create training dataset'}
        </Button>
        <Button
          size="sm"
          variant="secondary"
          onClick={() => void handleSplit()}
          loading={state.busy === 'split'}
          loadingLabel="Preparing…"
          disabled={busy || !state.hasExport}
          title={
            state.hasExport
              ? 'Create deterministic train/validation files from the training dataset'
              : 'Create the training dataset first'
          }
        >
          Prepare training split
        </Button>
      </div>
    </li>
  );
}

export function AdminTrainingPage() {
  const { state: courses } = useCourses();

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        title="Training"
        eyebrow="Admin"
        description="Build a training dataset from approved examples, then split it into train and validation sets."
      />

      <Callout tone="info" title="These two steps are real; job orchestration is not">
        <strong>Create training dataset</strong> exports every approved example
        for a course and validates it. <strong>Prepare train/validation
        split</strong> then divides that dataset deterministically. Both write
        real files on the backend. Everything after that — submitting a training
        job, tracking it, promoting the result — still happens through the
        scripts in <code>training/</code>. There is no endpoint for any of it,
        and no dataset versioning, so nothing here pretends otherwise.
      </Callout>

      <section className="ui-stack">
        <SectionHeader title="Per-course datasets" divider />

        {courses.status === 'loading' && (
          <p className="ui-text-muted" role="status" aria-live="polite">
            Loading courses…
          </p>
        )}

        {courses.status === 'error' && (
          <Callout tone="danger" title="Could not read courses">
            {courses.message}
          </Callout>
        )}

        {courses.status === 'ready' && (
          <ul className="admin-rows" aria-label="Course datasets">
            {courses.courses.map(({ courseId, metadata }) => (
              <CourseTrainingRow
                key={courseId}
                courseId={courseId}
                name={formatCourseHeading(metadata.name, metadata.title)}
              />
            ))}
          </ul>
        )}
      </section>

      <section className="ui-stack ui-stack--snug">
        <SectionHeader
          title="Training jobs"
          description="What this page will show once the backend can report it."
          divider
        />
        <EmptyState
          title="No job tracking yet"
          description="Submitting a run, watching its progress, and promoting the result are handled outside this application. See docs/frontend-backend-gaps.md for the endpoints this needs."
        />
      </section>
    </div>
  );
}
