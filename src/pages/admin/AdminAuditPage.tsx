import { useCallback, useEffect, useState } from 'react';
import { Button } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { PageHeader } from '../../components/ui/PageHeader';
import { listAudit, type AuditAction } from '../../lib/adminPeopleApi';

type State =
  | { status: 'loading' }
  | { status: 'ready'; actions: AuditAction[]; exhausted: boolean }
  | { status: 'error'; message: string };

const PAGE = 100;

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
  } catch {
    return iso;
  }
}

/**
 * The audit trail: every privileged action, newest first.
 *
 * Read-only. Rows are written by the backend as the actions happen and are
 * never edited; what is shown here is exactly what `admin_actions` holds,
 * minus anything credential-shaped, which the backend never writes.
 */
export function AdminAuditPage() {
  const [state, setState] = useState<State>({ status: 'loading' });
  const [loadingMore, setLoadingMore] = useState(false);

  const load = useCallback(async () => {
    setState({ status: 'loading' });
    try {
      const response = await listAudit(PAGE);
      setState({
        status: 'ready',
        actions: response.actions,
        exhausted: response.actions.length < PAGE,
      });
    } catch (error) {
      setState({
        status: 'error',
        message: error instanceof Error ? error.message : 'Could not load the audit trail.',
      });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function loadMore() {
    if (state.status !== 'ready' || state.actions.length === 0) {
      return;
    }
    setLoadingMore(true);
    try {
      const last = state.actions[state.actions.length - 1].actionId;
      const response = await listAudit(PAGE, last);
      setState({
        status: 'ready',
        actions: [...state.actions, ...response.actions],
        exhausted: response.actions.length < PAGE,
      });
    } catch (error) {
      setState({
        status: 'error',
        message: error instanceof Error ? error.message : 'Could not load the audit trail.',
      });
    } finally {
      setLoadingMore(false);
    }
  }

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        eyebrow="Admin"
        title="Audit"
        description="Privileged actions: invitations, role and course changes, disabled accounts, deleted research data."
        actions={
          <Button variant="secondary" onClick={() => void load()} iconLeft="status">
            Refresh
          </Button>
        }
      />

      {state.status === 'loading' && (
        <p className="ui-text-muted" role="status" aria-live="polite">
          Loading…
        </p>
      )}
      {state.status === 'error' && (
        <Callout tone="danger" title="Could not load the audit trail">
          {state.message}
        </Callout>
      )}
      {state.status === 'ready' && state.actions.length === 0 && (
        <p className="ui-text-muted">Nothing has been recorded yet.</p>
      )}
      {state.status === 'ready' && state.actions.length > 0 && (
        <>
          <ul className="admin-rows audit" aria-label="Audit trail">
            {state.actions.map((action) => (
              <li key={action.actionId} className="admin-row admin-row--stacked">
                <div className="admin-row__main">
                  <p className="admin-row__label">
                    <code>{action.action}</code>
                  </p>
                  <p className="admin-row__value">
                    {formatDate(action.createdAt)} ·{' '}
                    {action.actorName
                      ? `${action.actorName}${action.actorRole ? ` (${action.actorRole})` : ''}`
                      : action.actorRole === 'operator'
                        ? 'operator (command line)'
                        : 'unknown actor'}
                    {action.targetKind
                      ? ` · ${action.targetKind}${action.targetId ? ` ${action.targetId}` : ''}`
                      : ''}
                    {action.courseId ? ` · course ${action.courseId}` : ''}
                  </p>
                  {action.detail && Object.keys(action.detail).length > 0 && (
                    <pre className="admin-json audit__detail">
                      {JSON.stringify(action.detail)}
                    </pre>
                  )}
                </div>
              </li>
            ))}
          </ul>
          {!state.exhausted && (
            <div>
              <Button
                variant="secondary"
                onClick={() => void loadMore()}
                loading={loadingMore}
                loadingLabel="Loading…"
              >
                Load older
              </Button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
