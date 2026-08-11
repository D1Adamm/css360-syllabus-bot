import { useEffect, useState } from 'react';
import { Callout } from '../../components/ui/Callout';
import { EmptyState } from '../../components/ui/EmptyState';
import { PageHeader } from '../../components/ui/PageHeader';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { StatusPill } from '../../components/ui/StatusPill';
import { fetchFineTunedHealth, type FineTunedHealth } from '../../lib/adminApi';

/**
 * Model deployment state.
 *
 * The only thing the backend can currently answer is "what is the one shared
 * fine-tuned service reporting right now". There is no model registry, no
 * version history, no promotion record and no rollback, so this page shows the
 * live service and then says explicitly what is missing rather than rendering
 * an empty table that implies those features exist.
 */
export function AdminModelsPage() {
  const [health, setHealth] = useState<FineTunedHealth | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;

    void fetchFineTunedHealth()
      .then((result) => {
        if (!cancelled) {
          setHealth(result);
          setFailed(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setFailed(true);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        title="Models"
        eyebrow="Admin"
        description="The deployed fine-tuned inference service."
      />

      <section className="ui-stack">
        <SectionHeader title="Active service" divider />

        {failed && (
          <Callout tone="warning" title="Service did not respond">
            The fine-tuned inference service could not be reached.
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

      <section className="ui-stack">
        <SectionHeader title="Version history" divider />
        <EmptyState
          title="No model registry yet"
          description="Adapter versions, promotion, and rollback are handled by scripts outside this application. There is no endpoint for them to read from."
        />
      </section>
    </div>
  );
}
