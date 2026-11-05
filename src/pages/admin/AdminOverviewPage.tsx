import { useCallback, useEffect, useState } from 'react';
import { Button } from '../../components/ui/Button';
import { PageHeader } from '../../components/ui/PageHeader';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { StatusPill } from '../../components/ui/StatusPill';
import { useCourses } from '../../hooks/useCourses';
import {
  fetchBackendHealth,
  fetchFineTunedHealth,
  fetchModelReadiness,
  fetchStarterGenerationStatus,
  getConfiguredApiBaseUrl,
  type ModelReadiness,
  type ModelReadinessCourse,
} from '../../lib/adminApi';

type Probe =
  | { status: 'checking' }
  | { status: 'ok'; detail: string }
  | { status: 'degraded'; detail: string }
  | { status: 'down'; detail: string }
  /* We could not perform the check, so we know nothing about this service. */
  | { status: 'unknown'; detail: string };

interface Probes {
  backend: Probe;
  fineTuned: Probe;
  starter: Probe;
}

const INITIAL: Probes = {
  backend: { status: 'checking' },
  fineTuned: { status: 'checking' },
  starter: { status: 'checking' },
};

function pillTone(probe: Probe) {
  switch (probe.status) {
    case 'ok':
      return 'success' as const;
    case 'degraded':
      return 'warning' as const;
    case 'down':
      return 'danger' as const;
    case 'unknown':
      return 'neutral' as const;
    default:
      return 'progress' as const;
  }
}

function pillLabel(probe: Probe): string {
  switch (probe.status) {
    case 'ok':
      return 'ok';
    case 'degraded':
      return 'degraded';
    case 'down':
      return 'offline';
    case 'unknown':
      return 'not checked';
    default:
      return 'checking';
  }
}

/**
 * Per-course readiness, from the backend's assessment.
 *
 * `checking` while the request is out; `unknown` when it could not be made
 * (backend down, or an older backend without the route); otherwise the
 * assessment itself.
 */
type Readiness =
  | { status: 'checking' }
  | { status: 'unknown'; detail: string }
  | { status: 'ready'; readiness: ModelReadiness };

function courseTone(course: ModelReadinessCourse) {
  switch (course.problem) {
    case null:
      return 'success' as const;
    case 'no_activated_version':
      return 'neutral' as const;
    case 'service_unreachable':
    case 'activated_version_not_servable':
      return 'danger' as const;
    default:
      return 'warning' as const;
  }
}

function courseLabel(course: ModelReadinessCourse): string {
  switch (course.problem) {
    case null:
      return 'serving';
    case 'no_activated_version':
      return 'not activated';
    case 'service_unreachable':
      return 'service down';
    case 'activated_version_not_servable':
      return 'not loaded';
    default:
      return String(course.problem);
  }
}

/** What an operator reads on the row: activated vs newest, and the problem. */
function courseDetail(course: ModelReadinessCourse): string {
  const parts: string[] = [];
  parts.push(
    course.activatedVersion
      ? `active ${course.activatedVersion}`
      : 'no activated version',
  );
  if (course.newestVersion && course.newestVersion !== course.activatedVersion) {
    parts.push(
      `newest ${course.newestVersion}${course.newestStatus ? ` (${course.newestStatus})` : ''}`,
    );
  }
  if (course.servableVersions.length > 0) {
    parts.push(`service lists ${course.servableVersions.join(', ')}`);
  }
  if (course.message) {
    parts.push(course.message);
  }
  return parts.join(' · ');
}

export function AdminOverviewPage() {
  const [probes, setProbes] = useState<Probes>(INITIAL);
  const [readiness, setReadiness] = useState<Readiness>({ status: 'checking' });
  const [checkedAt, setCheckedAt] = useState<string | null>(null);
  const { state: courses } = useCourses();

  const runProbes = useCallback(async () => {
    setProbes(INITIAL);
    setReadiness({ status: 'checking' });

    const [backend, fineTuned, starter, models] = await Promise.allSettled([
      fetchBackendHealth(),
      fetchFineTunedHealth(),
      fetchStarterGenerationStatus(),
      fetchModelReadiness(),
    ]);

    /*
     * The downstream checks are proxied through the backend. If the backend
     * itself is offline, those requests failed because they never arrived —
     * not because the fine-tuned service or the generation worker is down.
     * Reporting them as "offline" would send an operator chasing a service
     * that may be perfectly healthy.
     */
    const backendUp = backend.status === 'fulfilled';
    const notChecked: Probe = {
      status: 'unknown',
      detail: 'Not checked — backend unavailable',
    };

    setProbes({
      backend: backendUp
        ? { status: 'ok', detail: backend.value.service }
        : { status: 'down', detail: 'No response from the API' },
      fineTuned: !backendUp
        ? notChecked
        : fineTuned.status === 'fulfilled'
          ? fineTuned.value.status === 'ok' && fineTuned.value.adapterLoaded
            ? {
                status: 'ok',
                detail: `${fineTuned.value.model ?? 'model'} · adapter loaded`,
              }
            : {
                status: 'degraded',
                detail: `status: ${fineTuned.value.status}, adapterLoaded: ${String(
                  fineTuned.value.adapterLoaded ?? 'unknown',
                )}`,
              }
          : { status: 'down', detail: 'No response from the service' },
      starter: !backendUp
        ? notChecked
        : starter.status === 'fulfilled'
          ? starter.value.active
            ? {
                status: 'degraded',
                detail: `${starter.value.operation ?? 'job'} running for ${
                  starter.value.courseId ?? 'unknown course'
                }`,
              }
            : { status: 'ok', detail: 'idle — no active job' }
          : { status: 'down', detail: 'No response from the API' },
    });

    setReadiness(
      !backendUp
        ? { status: 'unknown', detail: 'Not checked — backend unavailable' }
        : models.status === 'fulfilled'
          ? { status: 'ready', readiness: models.value }
          : {
              status: 'unknown',
              detail:
                models.reason instanceof Error
                  ? models.reason.message
                  : 'The readiness check did not answer',
            },
    );

    setCheckedAt(new Date().toLocaleTimeString());
  }, []);

  useEffect(() => {
    void runProbes();
  }, [runProbes]);

  const rows: { key: keyof Probes; label: string }[] = [
    { key: 'backend', label: 'Backend API' },
    { key: 'fineTuned', label: 'Fine-tuned service' },
    { key: 'starter', label: 'Seed generation' },
  ];

  const courseCount = courses.status === 'ready' ? courses.courses.length : null;
  const unindexed =
    courses.status === 'ready'
      ? courses.courses.filter(
          ({ metadata }) =>
            metadata.syllabusStatus !== 'indexed' && metadata.syllabusStatus !== 'ready',
        ).length
      : null;

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        title="Overview"
        eyebrow="Admin"
        description="Live status of the services this application depends on."
        actions={
          <Button variant="secondary" onClick={() => void runProbes()} iconLeft="status">
            Re-check
          </Button>
        }
      />

      <section className="ui-stack">
        <SectionHeader
          title="Services"
          description={checkedAt ? `Last checked at ${checkedAt}` : undefined}
          divider
        />
        <ul className="admin-rows" aria-label="Service status">
          {rows.map(({ key, label }) => {
            const probe = probes[key];
            return (
              <li key={key} className="admin-row">
                <span className="admin-row__label">{label}</span>
                <span className="admin-row__value">
                  {probe.status === 'checking' ? '…' : probe.detail}
                </span>
                <StatusPill tone={pillTone(probe)}>{pillLabel(probe)}</StatusPill>
              </li>
            );
          })}
        </ul>
        <p className="ui-text-xs ui-text-muted">
          The backend health endpoint reports only that the API is responding; it
          does not probe the local model runtime or the database. The fine-tuned
          and generation checks are proxied through the backend, so when it is
          offline they read &ldquo;not checked&rdquo; rather than offline.
          Course metadata is read from PostgreSQL through the backend, so it is
          unavailable for the same reasons the backend is.
        </p>
      </section>

      <section className="ui-stack">
        <SectionHeader
          title="Course models"
          description="Each registered course against what the fine-tuned service can serve. A course whose activated version is not loaded gets no Fine-Tuned answers; the backend logs the same check every few minutes."
          divider
        />
        {readiness.status === 'checking' ? (
          <p className="ui-text-muted" role="status" aria-live="polite">
            Checking course models…
          </p>
        ) : readiness.status === 'unknown' ? (
          <p className="ui-text-muted" role="status">
            {readiness.detail}
          </p>
        ) : (
          <>
            <ul className="admin-rows" aria-label="Course model readiness">
              <li className="admin-row">
                <span className="admin-row__label">Fine-tuned service</span>
                <span className="admin-row__value">
                  {readiness.readiness.service.reachable
                    ? `${readiness.readiness.mode} mode · status ${
                        readiness.readiness.service.status ?? 'unknown'
                      }`
                    : readiness.readiness.service.error ?? 'Not reachable'}
                </span>
                <StatusPill
                  tone={
                    readiness.readiness.service.ok
                      ? 'success'
                      : readiness.readiness.service.reachable
                        ? 'warning'
                        : 'danger'
                  }
                >
                  {readiness.readiness.service.ok
                    ? 'ok'
                    : readiness.readiness.service.reachable
                      ? readiness.readiness.service.status ?? 'degraded'
                      : 'offline'}
                </StatusPill>
              </li>
              {readiness.readiness.courses.map((course) => (
                <li key={course.courseId} className="admin-row">
                  <span className="admin-row__label">
                    {course.courseName ?? course.courseId}
                  </span>
                  <span className="admin-row__value">{courseDetail(course)}</span>
                  <StatusPill tone={courseTone(course)}>{courseLabel(course)}</StatusPill>
                </li>
              ))}
              {readiness.readiness.courses.length === 0 && (
                <li className="admin-row">
                  <span className="admin-row__value">No course has a registered model.</span>
                </li>
              )}
            </ul>
            {readiness.readiness.problems.length > 0 && (
              <p className="admin-row__error" role="alert">
                {readiness.readiness.problems.length} course
                {readiness.readiness.problems.length === 1 ? '' : 's'} cannot get Fine-Tuned
                answers right now. Fix the service or activate a version the service has
                loaded (Admin → Models).
              </p>
            )}
          </>
        )}
      </section>

      <section className="ui-stack">
        <SectionHeader title="Content" divider />
        <ul className="admin-rows" aria-label="Content summary">
          <li className="admin-row">
            <span className="admin-row__label">Courses</span>
            <span className="admin-row__value">
              {courseCount === null ? 'loading…' : `${courseCount} total`}
              {unindexed !== null && unindexed > 0
                ? ` · ${unindexed} without a prepared syllabus`
                : ''}
            </span>
          </li>
          <li className="admin-row">
            <span className="admin-row__label">API base URL</span>
            <span className="admin-row__value">
              <code>{getConfiguredApiBaseUrl() ?? 'not configured'}</code>
            </span>
          </li>
        </ul>
      </section>
    </div>
  );
}
