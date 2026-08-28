import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Callout } from '../../components/ui/Callout';
import { EmptyState } from '../../components/ui/EmptyState';
import { PageHeader } from '../../components/ui/PageHeader';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { StatusPill } from '../../components/ui/StatusPill';
import { useCourses } from '../../hooks/useCourses';
import { fetchFineTunedHealth, type FineTunedHealth } from '../../lib/adminApi';
import { formatCourseHeading } from '../../lib/courseLabels';
import {
  fetchCourseModel,
  getCurrentVersion,
  sortVersionsNewestFirst,
} from '../../lib/courseModelDb';
import { adminCoursePath } from '../../lib/roleRoutes';
import type { CourseModelRegistry, CourseModelVersion } from '../../types';

/**
 * Registered models per course, and the service that serves them.
 *
 * These are two different questions and the page keeps them in two sections.
 * A course's model exists because training produced an artifact and someone
 * registered it; that record is durable and unaffected by whether the shared
 * inference service happens to be up. The service check below says only whether
 * *something* is currently loaded — it is never used to decide whether a course
 * has a model.
 */

interface CourseRegistryRow {
  courseId: string;
  name: string;
  registry: CourseModelRegistry | null;
  failed: boolean;
}

function statusTone(version: CourseModelVersion) {
  switch (version.status) {
    case 'ready':
      return 'success' as const;
    case 'training':
      return 'progress' as const;
    case 'failed':
      return 'danger' as const;
    default:
      return 'neutral' as const;
  }
}

function deploymentTone(version: CourseModelVersion) {
  switch (version.deployment) {
    case 'online':
      return 'accent' as const;
    case 'offline':
      return 'warning' as const;
    default:
      return 'neutral' as const;
  }
}

export function AdminModelsPage() {
  const { state: courses } = useCourses();
  const [rows, setRows] = useState<CourseRegistryRow[] | null>(null);

  const [health, setHealth] = useState<FineTunedHealth | null>(null);
  const [healthFailed, setHealthFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;

    void fetchFineTunedHealth()
      .then((result) => {
        if (!cancelled) {
          setHealth(result);
          setHealthFailed(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setHealthFailed(true);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (courses.status !== 'ready') {
      return;
    }

    let cancelled = false;

    void Promise.all(
      courses.courses.map(async ({ courseId, metadata }) => {
        try {
          const registry = await fetchCourseModel(courseId);
          return {
            courseId,
            name: formatCourseHeading(metadata.name, metadata.title),
            registry,
            failed: false,
          };
        } catch {
          return {
            courseId,
            name: formatCourseHeading(metadata.name, metadata.title),
            registry: null,
            failed: true,
          };
        }
      }),
    ).then((result) => {
      if (!cancelled) {
        setRows(result);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [courses]);

  const registered = rows?.filter((row) => row.registry) ?? [];
  const unregistered = rows?.filter((row) => !row.registry && !row.failed) ?? [];

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        title="Models"
        eyebrow="Admin"
        description="Registered course models and the inference service that serves them."
      />

      <section className="ui-stack ui-stack--snug">
        <SectionHeader
          title="Registered course models"
          description="From each course's own registry record. Independent of service state."
          divider
        />

        {rows === null ? (
          <p className="ui-text-muted" role="status" aria-live="polite">
            Reading course registries…
          </p>
        ) : registered.length === 0 ? (
          <EmptyState
            title="No course models registered"
            description="A record is written when a trained adapter is promoted for a course."
          />
        ) : (
          <ul className="admin-rows" aria-label="Registered course models">
            {registered.map((row) => {
              const registry = row.registry!;
              const current = getCurrentVersion(registry);
              const history = sortVersionsNewestFirst(registry.versions);

              return (
                <li key={row.courseId} className="admin-row admin-row--stacked">
                  <div className="admin-row__main">
                    <Link
                      to={adminCoursePath(row.courseId)}
                      className="admin-row__label admin-row__label--link"
                    >
                      {row.name}
                    </Link>
                    <p className="admin-row__value">
                      <code>{row.courseId}</code>
                    </p>

                    {current && (
                      <p className="ui-text-xs ui-text-muted">
                        current <code>{current.version}</code> · base{' '}
                        <code>{current.baseModel}</code> ·{' '}
                        {current.trainingExampleCount} train examples · artifact{' '}
                        <code>{current.artifactRef}</code>
                      </p>
                    )}

                    {history.length > 1 && (
                      <details className="admin-probe">
                        <summary>Version history ({history.length})</summary>
                        <ul className="admin-chunks">
                          {history.map((version) => (
                            <li key={version.version}>
                              <code>{version.version}</code> · {version.status} ·{' '}
                              {version.trainingExampleCount} train examples ·{' '}
                              {new Date(version.createdAt).toLocaleDateString()}
                              {version.version === registry.currentVersion
                                ? ' · current'
                                : ''}
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                  </div>

                  {current && (
                    <div className="admin-row__actions">
                      <StatusPill tone={statusTone(current)}>{current.status}</StatusPill>
                      <StatusPill tone={deploymentTone(current)}>
                        {current.deployment}
                      </StatusPill>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}

        {unregistered.length > 0 && (
          <p className="ui-text-xs ui-text-muted">
            No model registered for: {unregistered.map((row) => row.courseId).join(', ')}
          </p>
        )}
      </section>

      <section className="ui-stack ui-stack--snug">
        <SectionHeader
          title="Inference service"
          description="One shared service. It reports what is loaded, not which course it belongs to."
          divider
        />

        {healthFailed && (
          <Callout tone="warning" title="Service did not respond">
            The fine-tuned inference service could not be reached. Registered
            models above are unaffected — this says nothing about whether they
            exist.
          </Callout>
        )}

        {health && (
          <ul className="admin-rows" aria-label="Fine-tuned service">
            <li className="admin-row">
              <span className="admin-row__label">Status</span>
              <span className="admin-row__value">{health.status}</span>
              <StatusPill tone={health.status === 'ok' ? 'success' : 'warning'}>
                {health.status}
              </StatusPill>
            </li>
            <li className="admin-row">
              <span className="admin-row__label">Model</span>
              <span className="admin-row__value">{health.model ?? '—'}</span>
            </li>
            <li className="admin-row">
              <span className="admin-row__label">Adapter loaded</span>
              <span className="admin-row__value">
                {String(health.adapterLoaded ?? 'unknown')}
              </span>
            </li>
            <li className="admin-row">
              <span className="admin-row__label">Service URL</span>
              <span className="admin-row__value">
                <code>{health.serviceUrl ?? '—'}</code>
              </span>
            </li>
            <li className="admin-row">
              <span className="admin-row__label">Host</span>
              <span className="admin-row__value">
                <code>
                  {health.hostname ?? '—'}
                  {health.port ? `:${health.port}` : ''}
                </code>
              </span>
            </li>
          </ul>
        )}
      </section>

      <Callout tone="info" title="Registered is not published">
        A successful training run registers its version automatically, as{' '}
        <code>ready</code> and <code>not published</code>. Publishing it to the
        cluster is a separate, deliberate step —{' '}
        <code>training/promote_qlora_adapter.sh</code> — and until then inference
        keeps answering from the previously published version.{' '}
        <code>scripts/register_course_model.py</code> remains as a recovery tool
        for an artifact that was produced but never reported.
      </Callout>
    </div>
  );
}
