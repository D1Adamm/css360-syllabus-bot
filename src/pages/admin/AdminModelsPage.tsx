import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Button } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { ConfirmDialog } from '../../components/ui/ConfirmDialog';
import { EmptyState } from '../../components/ui/EmptyState';
import { PageHeader } from '../../components/ui/PageHeader';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { StatusPill } from '../../components/ui/StatusPill';
import { useCourses } from '../../hooks/useCourses';
import { fetchFineTunedHealth, type FineTunedHealth } from '../../lib/adminApi';
import { formatCourseHeading } from '../../lib/courseLabels';
import {
  activateCourseModelVersion,
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

/**
 * The version a course's fine-tuned answers use: the activated one.
 *
 * Mirrors the backend's rule — the highest `online` version that is `ready`.
 * With none, the course has no Fine-Tuned answers at all; the newest
 * registered version is never used in its place.
 */
function activeVersion(registry: CourseModelRegistry): string | null {
  const online = Object.values(registry.versions)
    .filter((version) => version.deployment === 'online' && version.status === 'ready')
    .map((version) => version.version)
    .sort((left, right) => Number(left.slice(1)) - Number(right.slice(1)));
  return online.length > 0 ? online[online.length - 1]! : null;
}

/**
 * The versions the fine-tuned service says it can serve for a course, or null
 * when its health is not known. A hint only: the backend asks the service
 * again before activating anything.
 */
function servableVersions(
  health: FineTunedHealth | null,
  courseId: string,
): string[] | null {
  if (!health) {
    return null;
  }
  const entry = (health.courses ?? []).find((course) => course.courseId === courseId);
  return entry?.versions ?? [];
}

interface PendingActivation {
  courseId: string;
  name: string;
  version: string;
  previous: string | null;
}

export function AdminModelsPage() {
  const { state: courses } = useCourses();
  const [rows, setRows] = useState<CourseRegistryRow[] | null>(null);
  const [pending, setPending] = useState<PendingActivation | null>(null);
  const [activating, setActivating] = useState<string | null>(null);
  const [messages, setMessages] = useState<
    Record<string, { text: string; error: boolean }>
  >({});

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

  /**
   * Activates after confirmation. The backend checks the fine-tuned service
   * first and writes nothing on refusal; its explanation is shown as is.
   */
  async function activate(target: PendingActivation) {
    setPending(null);
    setActivating(target.courseId);
    setMessages((current) => {
      const next = { ...current };
      delete next[target.courseId];
      return next;
    });

    try {
      const result = await activateCourseModelVersion(target.courseId, target.version);
      setRows((current) =>
        (current ?? []).map((row) =>
          row.courseId === target.courseId ? { ...row, registry: result.registry } : row,
        ),
      );
      setMessages((current) => ({
        ...current,
        [target.courseId]: {
          text: result.unchanged
            ? `${result.version} was already active.`
            : `${result.version} is now active${
                result.previousVersion ? `, replacing ${result.previousVersion}` : ''
              }.`,
          error: false,
        },
      }));
    } catch (error) {
      setMessages((current) => ({
        ...current,
        [target.courseId]: {
          text: error instanceof Error ? error.message : 'Activation failed.',
          error: true,
        },
      }));
    } finally {
      setActivating(null);
    }
  }

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
              const active = activeVersion(registry);
              const servable = servableVersions(health, row.courseId);
              const message = messages[row.courseId];

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

                    <p className="ui-text-xs ui-text-muted">
                      {active ? (
                        <>
                          Active <code>{active}</code>: students&apos; Fine-Tuned answers
                          use this version.
                        </>
                      ) : (
                        <>
                          No activated version: Fine-Tuned and Fine-Tuned + RAG are
                          unavailable for this course until one is activated.
                        </>
                      )}
                    </p>

                    <ul
                      className="admin-chunks"
                      aria-label={`Model versions for ${row.courseId}`}
                    >
                      {history.map((version) => {
                        const isActive = version.version === active;
                        const onVm = servable === null
                          ? 'VM: unknown'
                          : servable.includes(version.version)
                            ? 'VM: servable'
                            : 'VM: not mapped';
                        const canActivate = version.status === 'ready' && !isActive;
                        const blocked =
                          servable !== null && !servable.includes(version.version);
                        return (
                          <li key={version.version}>
                            <code>{version.version}</code> · {version.status} ·{' '}
                            {isActive ? 'active' : 'not active'} · {onVm} ·{' '}
                            {version.trainingExampleCount} train examples ·{' '}
                            {new Date(version.createdAt).toLocaleDateString()}
                            {version.version === registry.currentVersion
                              ? ' · newest'
                              : ''}
                            {canActivate && (
                              <>
                                {' '}
                                <Button
                                  size="sm"
                                  variant="secondary"
                                  onClick={() =>
                                    setPending({
                                      courseId: row.courseId,
                                      name: row.name,
                                      version: version.version,
                                      previous: active,
                                    })
                                  }
                                  loading={activating === row.courseId}
                                  loadingLabel="Activating…"
                                  disabled={activating !== null || blocked}
                                  aria-label={`Activate ${version.version} for ${row.courseId}`}
                                  title={
                                    blocked
                                      ? 'The fine-tuned service cannot serve this version. Install and map it on the VM first.'
                                      : 'Make this the version students are answered by.'
                                  }
                                >
                                  Activate
                                </Button>
                              </>
                            )}
                          </li>
                        );
                      })}
                    </ul>

                    {message && (
                      <p
                        className={`ui-text-xs ${message.error ? 'admin-row__error' : 'ui-text-muted'}`}
                        role={message.error ? 'alert' : 'status'}
                      >
                        {message.text}
                      </p>
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
            {/*
              * Where the service runs is deliberately not shown, and not sent:
              * a compute node and a tunnel URL describe how to reach a machine,
              * and this page is served over a route that needs no credential.
              * What replaces them is what an operator can actually act on —
              * which courses the running service can answer for, and how much
              * of its allocation is left.
              */}
            <li className="admin-row">
              <span className="admin-row__label">Serving</span>
              <span className="admin-row__value">
                {health.courses && health.courses.length > 0
                  ? health.courses
                      .map(
                        (course) =>
                          `${course.courseId ?? 'unknown'} ${
                            course.currentVersion ?? '?'
                          }`,
                      )
                      .join(' · ')
                  : 'No published adapters'}
              </span>
            </li>
            <li className="admin-row">
              <span className="admin-row__label">Allocation left</span>
              <span className="admin-row__value">
                {typeof health.secondsRemaining === 'number'
                  ? `${Math.max(0, Math.round(health.secondsRemaining / 60))} min`
                  : '—'}
              </span>
            </li>
          </ul>
        )}
      </section>

      <Callout tone="info" title="Registered is not active">
        A successful training run registers its version automatically, as{' '}
        <code>ready</code> and not active; students keep the active version,
        and a course with no active version has no Fine-Tuned answers. To
        switch them, install and map the new version on the VM (
        <code>scripts/install_finetuned_adapter.py</code>, then{' '}
        <code>scripts/aiswe_finetuned.sh set-mapping</code> and restart), then
        press <strong>Activate</strong> above. Activation asks the fine-tuned
        service first and refuses a version it cannot serve. To roll back,
        activate the earlier version. Tillicum&apos;s{' '}
        <code>training/promote_qlora_adapter.sh</code> still records a
        publication too, as the legacy path.{' '}
        <code>scripts/register_course_model.py</code> remains as a recovery tool
        for an artifact that was produced but never reported.
      </Callout>

      <ConfirmDialog
        open={pending !== null}
        title={pending ? `Activate ${pending.version} for ${pending.name}?` : 'Activate'}
        description={
          pending
            ? `Students' Fine-Tuned and Fine-Tuned + RAG answers for this course ` +
              `switch to ${pending.version} immediately` +
              (pending.previous ? `, replacing ${pending.previous}` : '') +
              '. The fine-tuned service is checked first, and nothing changes if ' +
              'it cannot serve this version.' +
              (pending.previous
                ? ` To roll back, activate ${pending.previous} again.`
                : '')
            : undefined
        }
        confirmLabel="Activate"
        cancelLabel="Cancel"
        tone="default"
        busy={activating !== null}
        onConfirm={() => {
          if (pending) {
            void activate(pending);
          }
        }}
        onCancel={() => setPending(null)}
      />
    </div>
  );
}
