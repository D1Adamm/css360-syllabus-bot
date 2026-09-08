import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Button, LinkButton } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { formatCourseCode } from '../../lib/courseLabels';
import { PageHeader } from '../../components/ui/PageHeader';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { StatusPill } from '../../components/ui/StatusPill';
import { useCourseId } from '../../context/CourseContext';
import { useCourseExampleCounts } from '../../hooks/useCourseExampleCounts';
import { useCourseMetadata } from '../../hooks/useCourseMetadata';
import { useCourseModel } from '../../hooks/useCourseModel';
import { useCourseModelRequest } from '../../hooks/useCourseModelRequest';
import { getCurrentVersion, sortVersionsNewestFirst } from '../../lib/courseModelDb';
import { summariseCourseModel } from '../../lib/modelStatus';
import {
  ApiError,
  fetchCourseChunks,
  runSeedQualityCheck,
  type CourseChunksResponse,
  type SeedQualityCheckResponse,
} from '../../lib/adminApi';
import { useFactInventoryProbe } from '../../hooks/useFactInventoryProbe';
import {
  adminCourseExamplesPath,
  adminCourseReviewPath,
  studentCoursePath,
} from '../../lib/roleRoutes';
import { StudentAccessPanel } from '../../components/invite/StudentAccessPanel';
import {
  addMembership,
  listUsers,
  removeMembership,
  type AdminUser,
} from '../../lib/adminPeopleApi';

type Probe<T> =
  | { status: 'idle' }
  | { status: 'running' }
  | { status: 'ok'; data: T }
  | { status: 'failed'; message: string };

function errorText(error: unknown): string {
  return error instanceof ApiError ? error.message : String(error);
}

/** "10:42" for a build that started at an ISO timestamp; nothing if unparseable. */
function clockTime(iso: string | null): string | null {
  if (!iso) {
    return null;
  }
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) {
    return null;
  }
  return parsed.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/** The first N chunks are enough to see what the index looks like. */
const CHUNK_PREVIEW_COUNT = 12;

type PeopleState =
  | { status: 'loading' }
  | { status: 'ready'; users: AdminUser[] }
  | { status: 'failed'; message: string };

/**
 * Who teaches this course, and who could.
 *
 * Memberships are the course-scoped half of authorization: a professor sees
 * and manages exactly the courses listed here against their name. Adding and
 * removing goes through the administrator-only routes and is audited.
 */
function InstructorsSection({ courseId }: { courseId: string }) {
  const [people, setPeople] = useState<PeopleState>({ status: 'loading' });
  const [choice, setChoice] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setPeople({ status: 'ready', users: (await listUsers()).users });
    } catch (caught) {
      setPeople({ status: 'failed', message: errorText(caught) });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function run(action: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await action();
      setChoice('');
      await load();
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setBusy(false);
    }
  }

  const users = people.status === 'ready' ? people.users : [];
  const instructors = users.filter(
    (user) => user.role === 'professor' && user.courseIds.includes(courseId),
  );
  const candidates = users.filter(
    (user) => user.role === 'professor' && !user.disabled && !user.courseIds.includes(courseId),
  );

  return (
    <section className="ui-stack ui-stack--snug">
      <SectionHeader
        title="Instructors"
        description="Professors assigned to this course. Administrators reach every course without being listed."
        divider
      />
      {people.status === 'loading' && (
        <p className="ui-text-muted" role="status" aria-live="polite">
          Loading instructors…
        </p>
      )}
      {people.status === 'failed' && (
        <Callout tone="danger" title="Could not load instructors">
          {people.message}
        </Callout>
      )}
      {error && (
        <Callout tone="danger" title="That did not work">
          {error}
        </Callout>
      )}
      {people.status === 'ready' && (
        <ul className="admin-rows" aria-label="Instructors">
          {instructors.length === 0 && (
            <li className="admin-row">
              <span className="admin-row__value ui-text-muted">
                No instructor is assigned yet.
              </span>
            </li>
          )}
          {instructors.map((user) => (
            <li key={user.userId} className="admin-row">
              <span className="admin-row__label">{user.displayName}</span>
              <span className="admin-row__value">{user.email}</span>
              <Button
                size="sm"
                variant="ghost"
                disabled={busy}
                onClick={() => void run(() => removeMembership(user.userId, courseId))}
              >
                Remove
              </Button>
            </li>
          ))}
          {candidates.length > 0 && (
            <li className="admin-row">
              <label className="ui-visually-hidden" htmlFor="assign-instructor">
                Assign an instructor
              </label>
              <select
                id="assign-instructor"
                className="ui-input ui-input--sm"
                value={choice}
                disabled={busy}
                onChange={(event) => setChoice(event.target.value)}
              >
                <option value="">Assign an instructor…</option>
                {candidates.map((user) => (
                  <option key={user.userId} value={user.userId}>
                    {user.displayName} · {user.email}
                  </option>
                ))}
              </select>
              <Button
                size="sm"
                variant="secondary"
                disabled={busy || !choice}
                onClick={() => void run(() => addMembership(choice, courseId))}
              >
                Add
              </Button>
            </li>
          )}
        </ul>
      )}
    </section>
  );
}

/**
 * Technical detail for one course.
 *
 * Runs the diagnostics the backend has always exposed but nothing ever called:
 * the chunk listing, the fact inventory, and the dataset quality report. All
 * three are read-only inspections — none of them generate seeds or mutate the
 * course.
 */
export function AdminCourseDetailPage() {
  const courseId = useCourseId();
  const { metadata } = useCourseMetadata(courseId);
  const countsState = useCourseExampleCounts(courseId);
  const { state: modelState } = useCourseModel(courseId);
  const { state: requestState } = useCourseModelRequest(courseId);

  const [chunks, setChunks] = useState<Probe<CourseChunksResponse>>({ status: 'idle' });
  /*
   * The fact inventory is the one diagnostic that can take longer than a page
   * should wait. Building it is every batch of the syllabus through the local
   * model on the CPU, and this control used to hold one request open for all
   * of it — "Building…" until somebody reloaded. The hook asks the backend for
   * status instead, polls for a bounded time, and always ends somewhere a
   * reader can act on: a result, an error with Retry, or "still running,
   * check again".
   */
  const { state: facts, inspect: inspectFacts } = useFactInventoryProbe(courseId);
  const [quality, setQuality] = useState<Probe<SeedQualityCheckResponse>>({
    status: 'idle',
  });

  const loadChunks = useCallback(async () => {
    setChunks({ status: 'running' });
    try {
      setChunks({ status: 'ok', data: await fetchCourseChunks(courseId) });
    } catch (error) {
      setChunks({ status: 'failed', message: errorText(error) });
    }
  }, [courseId]);

  // Inspect expands the listing in place; Collapse puts the row back the way
  // it was. Re-inspecting reads the file again, which is cheap.
  const toggleChunks = useCallback(() => {
    if (chunks.status === 'ok') {
      setChunks({ status: 'idle' });
      return;
    }
    void loadChunks();
  }, [chunks.status, loadChunks]);

  /*
   * Two sources report a chunk count, and they can disagree.
   *
   * The Record row is `courses.chunk_count` in PostgreSQL, written when the
   * syllabus was uploaded. The inspector counts the chunks in the index file
   * on disk, which is what retrieval actually reads. A rebuild with the
   * reindex script rewrites the file; until it also updated the record, a
   * course could show 92 in one place and 163 in the other. When they differ
   * the page says so and names the fix rather than showing one number.
   */
  const recordChunkCount = metadata?.chunkCount ?? null;
  const indexChunkCount = chunks.status === 'ok' ? chunks.data.chunkCount : null;
  const chunkCountMismatch =
    recordChunkCount !== null &&
    indexChunkCount !== null &&
    recordChunkCount !== indexChunkCount;
  const factsStartedAt =
    facts.status === 'building' || facts.status === 'pending'
      ? clockTime(facts.startedAt)
      : null;

  const loadQuality = useCallback(async () => {
    setQuality({ status: 'running' });
    try {
      setQuality({ status: 'ok', data: await runSeedQualityCheck(courseId) });
    } catch (error) {
      setQuality({ status: 'failed', message: errorText(error) });
    }
  }, [courseId]);

  const counts = countsState.status === 'ready' ? countsState.counts : null;

  /*
   * The registered model for this course, from the registry record itself.
   *
   * This section used to be a fixed sentence saying no per-course registry
   * existed. It has existed since the `course_models` tables landed, and Admin
   * Models, Admin Training and the professor Model page have all been reading
   * it — so this page alone said `unknown` about a course whose model was
   * current, ready and online. It now reads the same two course-scoped records
   * through `summariseCourseModel`, the same helper the professor overview
   * uses, so the four surfaces cannot disagree again.
   */
  const registry = modelState.status === 'ready' ? modelState.registry : null;
  const currentVersion = registry ? getCurrentVersion(registry) : null;
  const model = summariseCourseModel({
    version: currentVersion,
    request: requestState.status === 'ready' ? requestState.request : null,
    loading: modelState.status === 'loading' || requestState.status === 'loading',
    registryUnavailable: modelState.status === 'unavailable',
    requestUnavailable: requestState.status === 'unavailable',
  });
  const history = registry ? sortVersionsNewestFirst(registry.versions) : [];

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        eyebrow="Admin"
        title={formatCourseCode(metadata?.name) || courseId}
        description={metadata?.title}
        actions={
          /* This course's student pages as an administrator sees them: real
             answers from the same four routes, and nothing saved. The shell
             shows a preview banner there; see AdminPreviewContext. */
          <LinkButton
            to={studentCoursePath(courseId, 'compare')}
            variant="secondary"
            iconRight="forward"
            title="Open this course's Compare page the way a student sees it. Answers are real; nothing you submit is saved."
          >
            Preview Student Experience
          </LinkButton>
        }
      />

      <section className="ui-stack ui-stack--snug">
        <SectionHeader title="Record" divider />
        <ul className="admin-rows" aria-label="Course record">
          <li className="admin-row">
            <span className="admin-row__label">Course id</span>
            <span className="admin-row__value">
              <code>{courseId}</code>
            </span>
          </li>
          <li className="admin-row">
            <span className="admin-row__label">Term</span>
            <span className="admin-row__value">{metadata?.term ?? '—'}</span>
          </li>
          <li className="admin-row">
            <span className="admin-row__label">Syllabus status</span>
            <span className="admin-row__value">
              <code>{metadata?.syllabusStatus ?? 'unknown'}</code>
            </span>
          </li>
          <li className="admin-row">
            <span className="admin-row__label">Syllabus file</span>
            <span className="admin-row__value">
              {metadata?.syllabusFileName || '—'}{' '}
              {metadata?.syllabusType ? `(${metadata.syllabusType})` : ''}
            </span>
          </li>
          <li className="admin-row">
            <span className="admin-row__label">Index chunks (course record)</span>
            <span className="admin-row__value">
              {metadata?.chunkCount ?? 0}{' '}
              <span className="ui-text-xs ui-text-muted">
                as stored in PostgreSQL when the syllabus was indexed; the index file
                itself is under Diagnostics
              </span>
            </span>
          </li>
          <li className="admin-row">
            <span className="admin-row__label">Examples</span>
            <span className="admin-row__value">
              {counts
                ? `${counts.total} total · ${counts.approved} approved · ${counts.pending} pending · ${counts.rejected} rejected · ${counts.edited} edited`
                : countsState.status === 'loading'
                  ? 'loading…'
                  : 'unavailable'}
            </span>
            <Link to={adminCourseExamplesPath(courseId)} className="admin-row__label--link">
              Open dataset
            </Link>
            <Link to={adminCourseReviewPath(courseId)} className="admin-row__label--link">
              Review examples
            </Link>
          </li>
        </ul>
      </section>

      <section className="ui-stack ui-stack--snug">
        <SectionHeader
          title="Diagnostics"
          description="Read-only inspections. None of these generate examples or modify the course."
          divider
        />

        <ul className="admin-rows" aria-label="Diagnostics">
          <li className="admin-row admin-row--stacked">
            <div className="admin-row__main">
              <p className="admin-row__label">Syllabus index</p>
              <p className="ui-text-xs ui-text-muted">
                The index file on disk that retrieval reads, and the chunks it was built from.
              </p>
              {chunks.status === 'failed' && (
                <p className="admin-row__error" role="alert">
                  {chunks.message}
                </p>
              )}
              {chunks.status === 'ok' && (
                <div className="admin-probe">
                  <p className="admin-row__value">
                    {chunks.data.chunkCount} chunks in the index file
                    {chunks.data.documentTitle ? ` · ${chunks.data.documentTitle}` : ''}
                    {chunks.data.indexVersion != null
                      ? ` · index v${chunks.data.indexVersion}`
                      : ''}
                  </p>
                  {chunkCountMismatch && (
                    <Callout
                      tone="warning"
                      title="The course record disagrees with the index file"
                      live={false}
                    >
                      <p>
                        The record in PostgreSQL says {recordChunkCount} chunks; the index
                        file holds {indexChunkCount}. The record is written when a syllabus
                        is uploaded, and a rebuild with the reindex script replaced the file
                        without updating it. Retrieval uses the file, so {indexChunkCount} is
                        the real count.
                      </p>
                      <p>To bring the record up to date without re-embedding, run on the VM:</p>
                      <pre className="admin-json">
                        {`python -m app.reindex_course --course-id ${courseId} --sync-record`}
                      </pre>
                    </Callout>
                  )}
                  <ol className="admin-chunks">
                    {chunks.data.chunks.slice(0, CHUNK_PREVIEW_COUNT).map((chunk) => (
                      <li key={chunk.chunkId}>
                        <code>{chunk.chunkId}</code> {chunk.sectionTitle}
                      </li>
                    ))}
                  </ol>
                  {chunks.data.chunks.length > CHUNK_PREVIEW_COUNT && (
                    <p className="ui-text-xs ui-text-muted">
                      Showing the first {CHUNK_PREVIEW_COUNT} of {chunks.data.chunks.length}.
                    </p>
                  )}
                </div>
              )}
            </div>
            <div className="admin-row__actions">
              <Button
                size="sm"
                variant="secondary"
                onClick={toggleChunks}
                loading={chunks.status === 'running'}
                loadingLabel="Reading…"
                aria-expanded={chunks.status === 'ok'}
              >
                {chunks.status === 'ok'
                  ? 'Collapse'
                  : chunks.status === 'failed'
                    ? 'Retry'
                    : 'Inspect'}
              </Button>
            </div>
          </li>

          <li className="admin-row admin-row--stacked">
            <div className="admin-row__main">
              <p className="admin-row__label">Fact inventory</p>
              <p className="ui-text-xs ui-text-muted">
                Extraction only — builds or reuses the cached inventory. Does not
                generate seeds.
              </p>
              {facts.status === 'building' && (
                <p className="ui-text-xs ui-text-muted" role="status" aria-live="polite">
                  Building on the server{factsStartedAt ? ` since ${factsStartedAt}` : ''}…
                  Extraction runs every batch of the syllabus through the local model and
                  can take several minutes. This page checks every few seconds.
                </p>
              )}
              {facts.status === 'pending' && (
                <p className="ui-text-xs ui-text-muted" role="status" aria-live="polite">
                  Still building on the server{factsStartedAt ? ` since ${factsStartedAt}` : ''}.
                  This page stopped checking after two minutes; the build carries on
                  without it. Check again in a while.
                </p>
              )}
              {facts.status === 'failed' && (
                <p className="admin-row__error" role="alert">
                  {facts.message}
                </p>
              )}
              {facts.status === 'ok' && (
                <div className="admin-probe">
                  <p className="admin-row__value">
                    {facts.data.factCount} facts · model <code>{facts.data.model}</code>
                    {facts.data.cached ? ' · cached' : ''}
                    {facts.data.fallbackUsed ? ' · fallback used' : ''}
                  </p>
                  {facts.data.countsByKind &&
                    Object.keys(facts.data.countsByKind).length > 0 && (
                      <p className="ui-text-xs ui-text-muted">
                        {Object.entries(facts.data.countsByKind)
                          .map(([kind, count]) => `${kind}: ${count}`)
                          .join(' · ')}
                      </p>
                    )}
                </div>
              )}
            </div>
            <div className="admin-row__actions">
              <Button
                size="sm"
                variant="secondary"
                onClick={inspectFacts}
                loading={facts.status === 'building'}
                loadingLabel="Building…"
              >
                {facts.status === 'pending'
                  ? 'Check again'
                  : facts.status === 'failed'
                    ? 'Retry'
                    : 'Inspect'}
              </Button>
            </div>
          </li>

          <li className="admin-row admin-row--stacked">
            <div className="admin-row__main">
              <p className="admin-row__label">Dataset quality</p>
              <p className="ui-text-xs ui-text-muted">
                Runs the quality report over this course&apos;s stored examples.
              </p>
              {quality.status === 'failed' && (
                <p className="admin-row__error" role="alert">
                  {quality.message}
                </p>
              )}
              {quality.status === 'ok' && (
                <details className="admin-probe">
                  <summary>Report</summary>
                  <pre className="admin-json">
                    {JSON.stringify(quality.data.report, null, 2)}
                  </pre>
                </details>
              )}
            </div>
            <div className="admin-row__actions">
              <Button
                size="sm"
                variant="secondary"
                onClick={() => void loadQuality()}
                loading={quality.status === 'running'}
                loadingLabel="Checking…"
              >
                Run check
              </Button>
            </div>
          </li>
        </ul>
      </section>

      <InstructorsSection courseId={courseId} />

      <StudentAccessPanel courseId={courseId} courseName={metadata?.name} />

      <Callout tone="info" title="Seed generation is not exposed here">
        The generation endpoints exist but are long-running and CPU-bound, and
        firing one from a page that can be closed mid-run is a good way to leave
        a job orphaned. Run them from the backend until there is real job
        tracking.
      </Callout>

      <section className="ui-stack ui-stack--snug">
        <SectionHeader
          title="Model state"
          description="This course's registry record. Independent of whether anything is serving it right now."
          divider
        />
        <ul className="admin-rows" aria-label="Model state">
          <li className="admin-row">
            <span className="admin-row__label">Course model</span>
            <span className="admin-row__value">
              {currentVersion ? (
                <>
                  <code>{currentVersion.version}</code> · base{' '}
                  <code>{currentVersion.baseModel}</code> ·{' '}
                  {currentVersion.trainingExampleCount} train examples · artifact{' '}
                  <code>{currentVersion.artifactRef}</code>
                </>
              ) : modelState.status === 'loading' ? (
                'reading the registry…'
              ) : modelState.status === 'unavailable' ? (
                'The registry could not be read. This says nothing about whether a model exists.'
              ) : (
                'No model version is registered for this course.'
              )}
            </span>
            <StatusPill tone={model.tone}>{model.label}</StatusPill>
          </li>

          {currentVersion && (
            <li className="admin-row">
              <span className="admin-row__label">Version state</span>
              <span className="admin-row__value">
                status <code>{currentVersion.status}</code> · deployment{' '}
                <code>{currentVersion.deployment}</code>
                {currentVersion.runId ? (
                  <>
                    {' '}
                    · run <code>{currentVersion.runId}</code>
                  </>
                ) : null}
              </span>
            </li>
          )}

          {history.length > 1 && (
            <li className="admin-row admin-row--stacked">
              <div className="admin-row__main">
                <p className="admin-row__label">Version history</p>
                <ul className="admin-chunks">
                  {history.map((version) => (
                    <li key={version.version}>
                      <code>{version.version}</code> · {version.status} ·{' '}
                      {version.deployment} · {version.trainingExampleCount} train
                      examples
                      {registry && version.version === registry.currentVersion
                        ? ' · current'
                        : ''}
                    </li>
                  ))}
                </ul>
              </div>
            </li>
          )}
        </ul>
      </section>
    </div>
  );
}
