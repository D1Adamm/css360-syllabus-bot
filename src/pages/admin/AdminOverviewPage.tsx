import { useCallback, useEffect, useState } from 'react';
import { Button } from '../../components/ui/Button';
import { PageHeader } from '../../components/ui/PageHeader';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { StatusPill } from '../../components/ui/StatusPill';
import { useCourses } from '../../hooks/useCourses';
import {
  fetchBackendHealth,
  fetchFineTunedHealth,
  fetchStarterGenerationStatus,
  getConfiguredApiBaseUrl,
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

export function AdminOverviewPage() {
  const [probes, setProbes] = useState<Probes>(INITIAL);
  const [checkedAt, setCheckedAt] = useState<string | null>(null);
  const { state: courses } = useCourses();

  const runProbes = useCallback(async () => {
    setProbes(INITIAL);

    const [backend, fineTuned, starter] = await Promise.allSettled([
      fetchBackendHealth(),
      fetchFineTunedHealth(),
      fetchStarterGenerationStatus(),
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
