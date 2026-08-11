import { useCallback, useState } from 'react';
import { Link } from 'react-router-dom';
import { Button } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { formatCourseCode } from '../../lib/courseLabels';
import { PageHeader } from '../../components/ui/PageHeader';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { StatusPill } from '../../components/ui/StatusPill';
import { useCourseId } from '../../context/CourseContext';
import { useCourseExampleCounts } from '../../hooks/useCourseExampleCounts';
import { useCourseMetadata } from '../../hooks/useCourseMetadata';
import {
  ApiError,
  fetchCourseChunks,
  fetchFactInventory,
  runSeedQualityCheck,
  type CourseChunksResponse,
  type FactInventoryResponse,
  type SeedQualityCheckResponse,
} from '../../lib/adminApi';
import { adminCourseExamplesPath } from '../../lib/roleRoutes';

type Probe<T> =
  | { status: 'idle' }
  | { status: 'running' }
  | { status: 'ok'; data: T }
  | { status: 'failed'; message: string };

function errorText(error: unknown): string {
  return error instanceof ApiError ? error.message : String(error);
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

  const [chunks, setChunks] = useState<Probe<CourseChunksResponse>>({ status: 'idle' });
  const [facts, setFacts] = useState<Probe<FactInventoryResponse>>({ status: 'idle' });
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

  const loadFacts = useCallback(async () => {
    setFacts({ status: 'running' });
    try {
      setFacts({ status: 'ok', data: await fetchFactInventory(courseId) });
    } catch (error) {
      setFacts({ status: 'failed', message: errorText(error) });
    }
  }, [courseId]);

  const loadQuality = useCallback(async () => {
    setQuality({ status: 'running' });
    try {
      setQuality({ status: 'ok', data: await runSeedQualityCheck(courseId) });
    } catch (error) {
      setQuality({ status: 'failed', message: errorText(error) });
    }
  }, [courseId]);

  const counts = countsState.status === 'ready' ? countsState.counts : null;

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        eyebrow="Admin"
        title={formatCourseCode(metadata?.name) || courseId}
        description={metadata?.title}
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
            <span className="admin-row__label">Index chunks</span>
            <span className="admin-row__value">{metadata?.chunkCount ?? 0}</span>
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
                Chunks the retrieval index was built from.
              </p>
              {chunks.status === 'failed' && (
                <p className="admin-row__error" role="alert">
                  {chunks.message}
                </p>
              )}
              {chunks.status === 'ok' && (
                <div className="admin-probe">
                  <p className="admin-row__value">
                    {chunks.data.chunkCount} chunks
                    {chunks.data.documentTitle ? ` · ${chunks.data.documentTitle}` : ''}
                    {chunks.data.indexVersion != null
                      ? ` · index v${chunks.data.indexVersion}`
                      : ''}
                  </p>
                  <ol className="admin-chunks">
                    {chunks.data.chunks.slice(0, 12).map((chunk) => (
                      <li key={chunk.chunkId}>
                        <code>{chunk.chunkId}</code> {chunk.sectionTitle}
                      </li>
                    ))}
                  </ol>
                  {chunks.data.chunks.length > 12 && (
                    <p className="ui-text-xs ui-text-muted">
                      Showing the first 12 of {chunks.data.chunks.length}.
                    </p>
                  )}
                </div>
              )}
            </div>
            <div className="admin-row__actions">
              <Button
                size="sm"
                variant="secondary"
                onClick={() => void loadChunks()}
                loading={chunks.status === 'running'}
                loadingLabel="Reading…"
              >
                Inspect
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
                onClick={() => void loadFacts()}
                loading={facts.status === 'running'}
                loadingLabel="Building…"
              >
                Inspect
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

      <Callout tone="info" title="Seed generation is not exposed here">
        The generation endpoints exist but are long-running and CPU-bound, and
        firing one from a page that can be closed mid-run is a good way to leave
        a job orphaned. Run them from the backend until there is real job
        tracking.
      </Callout>

      <section className="ui-stack ui-stack--snug">
        <SectionHeader title="Model state" divider />
        <ul className="admin-rows" aria-label="Model state">
          <li className="admin-row">
            <span className="admin-row__label">Course model</span>
            <span className="admin-row__value">
              No per-course model registry exists, so this cannot be answered.
            </span>
            <StatusPill tone="neutral">unknown</StatusPill>
          </li>
        </ul>
      </section>
    </div>
  );
}
