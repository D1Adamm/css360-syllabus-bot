import { useCallback, useEffect, useState } from 'react';
import { Button } from '../ui/Button';
import { Callout } from '../ui/Callout';
import { ConfirmDialog } from '../ui/ConfirmDialog';
import { EmptyState } from '../ui/EmptyState';
import { SectionHeader } from '../ui/SectionHeader';
import { StatusPill } from '../ui/StatusPill';
import {
  createStudentInvite,
  joinLinkUrl,
  joinPageUrl,
  listStudentInvites,
  revokeStudentInvite,
  type StudentInvite,
} from '../../lib/inviteApi';
import { formatCourseCode } from '../../lib/courseLabels';

type ListState =
  | { status: 'loading' }
  | { status: 'ready'; invites: StudentInvite[] }
  | { status: 'error'; message: string };

function siteOrigin(): string {
  return typeof window === 'undefined' ? '' : window.location.origin;
}

function displayHost(origin: string): string {
  return origin.replace(/^https?:\/\//, '');
}

function formatDate(iso: string | null | undefined): string {
  if (!iso) {
    return '';
  }
  try {
    return new Date(iso).toLocaleDateString(undefined, { dateStyle: 'medium' });
  } catch {
    return iso;
  }
}

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

export interface StudentAccessPanelProps {
  courseId: string;
  /** Course code for the heading, e.g. "CSS 360". */
  courseName?: string | null;
}

/**
 * Classroom access for one course.
 *
 * What an instructor puts on the board: the join page and a six-character
 * code. The direct link and the QR code are conveniences for Canvas and for
 * phones; the code is the primary thing because the class is on laptops.
 *
 * Every action goes through the backend, which decides whether this staff
 * member may manage this course's codes. The panel is embedded on the
 * professor's Invite page and on the administrator's course page unchanged.
 */
export function StudentAccessPanel({ courseId, courseName }: StudentAccessPanelProps) {
  const [list, setList] = useState<ListState>({ status: 'loading' });
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [copied, setCopied] = useState<'link' | 'code' | 'page' | null>(null);
  const [qr, setQr] = useState<{ code: string; dataUrl: string } | null>(null);
  const [qrError, setQrError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<'revoke' | 'replace' | null>(null);

  const load = useCallback(async () => {
    setList({ status: 'loading' });
    try {
      const response = await listStudentInvites(courseId);
      setList({ status: 'ready', invites: response.invites });
    } catch (caught) {
      setList({
        status: 'error',
        message: caught instanceof Error ? caught.message : 'Could not load the class codes.',
      });
    }
  }, [courseId]);

  useEffect(() => {
    void load();
  }, [load]);

  const invites = list.status === 'ready' ? list.invites : [];
  const active = invites.find((invite) => invite.status === 'active') ?? null;
  const retired = invites.filter((invite) => invite.status !== 'active');
  const origin = siteOrigin();
  const pageUrl = joinPageUrl(origin);
  const link = active ? joinLinkUrl(origin, active.code) : null;

  async function run(action: () => Promise<unknown>) {
    setBusy(true);
    setActionError(null);
    try {
      await action();
      setQr(null);
      await load();
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : 'That did not work.');
    } finally {
      setBusy(false);
      setConfirming(null);
    }
  }

  async function copy(kind: 'link' | 'code' | 'page', text: string) {
    if (await copyText(text)) {
      setCopied(kind);
      window.setTimeout(() => setCopied(null), 2500);
    }
  }

  async function showQr() {
    if (!active || !link) {
      return;
    }
    setQrError(null);
    try {
      const { toDataURL } = await import('qrcode');
      const dataUrl = await toDataURL(link, { width: 240, margin: 1 });
      setQr({ code: active.code, dataUrl });
    } catch {
      setQrError('The QR code could not be drawn. The link and the code still work.');
    }
  }

  const heading = `${formatCourseCode(courseName) || 'Course'} student access`;

  return (
    <section className="ui-stack ui-stack--snug access" aria-label={heading}>
      <SectionHeader
        title="Student access"
        description="Students open the join page on their laptop and enter the class code. No account, no name, no email."
        divider
      />

      {list.status === 'loading' && (
        <p className="ui-text-muted" role="status" aria-live="polite">
          Loading class codes…
        </p>
      )}

      {list.status === 'error' && (
        <Callout
          tone="danger"
          title="Could not load class codes"
          actions={
            <Button size="sm" variant="secondary" onClick={() => void load()}>
              Try again
            </Button>
          }
        >
          {list.message}
        </Callout>
      )}

      {actionError && (
        <Callout tone="danger" title="That did not work">
          {actionError}
        </Callout>
      )}

      {list.status === 'ready' && !active && (
        <EmptyState
          illustration="contribute"
          title="No class code yet"
          description="Create one and put it on the board. Every student who enters it gets their own anonymous identity in this course."
          action={
            <Button
              variant="primary"
              iconLeft="add"
              loading={busy}
              loadingLabel="Creating…"
              onClick={() => void run(() => createStudentInvite(courseId))}
            >
              Create class code
            </Button>
          }
        />
      )}

      {list.status === 'ready' && active && link && (
        <div className="access__card">
          <div className="access__board">
            <div className="access__item">
              <span className="access__label">Join page</span>
              <span className="access__page">{displayHost(pageUrl)}</span>
            </div>
            <div className="access__item">
              <span className="access__label">Class code</span>
              <span className="access__code" aria-label={`Class code ${active.code.split('').join(' ')}`}>
                {active.code}
              </span>
            </div>
          </div>

          <p className="access__link">
            Direct link for Canvas or Discord: <code>{link}</code>
          </p>

          <div className="access__actions">
            <Button
              size="sm"
              variant="secondary"
              iconLeft="copy"
              onClick={() => void copy('link', link)}
            >
              {copied === 'link' ? 'Copied' : 'Copy link'}
            </Button>
            <Button
              size="sm"
              variant="secondary"
              iconLeft="copy"
              onClick={() => void copy('code', active.code)}
            >
              {copied === 'code' ? 'Copied' : 'Copy code'}
            </Button>
            <Button size="sm" variant="secondary" iconLeft="link" onClick={() => void showQr()}>
              Show QR
            </Button>
            <Button
              size="sm"
              variant="secondary"
              iconLeft="add"
              disabled={busy}
              onClick={() => setConfirming('replace')}
            >
              New code
            </Button>
            <Button
              size="sm"
              variant="danger"
              iconLeft="delete"
              disabled={busy}
              onClick={() => setConfirming('revoke')}
            >
              Revoke
            </Button>
          </div>

          {qrError && <p className="ui-text-xs ui-text-muted">{qrError}</p>}
          {qr && qr.code === active.code && (
            <figure className="access__qr">
              <img src={qr.dataUrl} alt={`QR code for ${link}`} width={240} height={240} />
              <figcaption className="ui-text-xs ui-text-muted">
                Scanning opens the join page with the code filled in.
              </figcaption>
            </figure>
          )}

          <p className="access__meta ui-text-xs ui-text-muted">
            {active.label ? `${active.label} · ` : ''}
            Created {formatDate(active.createdAt)}
            {active.createdByName ? ` by ${active.createdByName}` : ''} · used{' '}
            {active.useCount} {active.useCount === 1 ? 'time' : 'times'} ·{' '}
            {active.participantCount} {active.participantCount === 1 ? 'student' : 'students'}
            {active.expiresAt ? ` · expires ${formatDate(active.expiresAt)}` : ' · does not expire'}
          </p>
        </div>
      )}

      {retired.length > 0 && (
        <details className="access__history">
          <summary>
            {retired.length} earlier {retired.length === 1 ? 'code' : 'codes'}
          </summary>
          <ul className="access__history-list">
            {retired.map((invite) => (
              <li key={invite.invitationId} className="access__history-row">
                <code>{invite.code}</code>
                <StatusPill tone={invite.status === 'revoked' ? 'danger' : 'neutral'}>
                  {invite.status}
                </StatusPill>
                <span className="ui-text-xs ui-text-muted">
                  used {invite.useCount} {invite.useCount === 1 ? 'time' : 'times'}
                  {invite.revokedAt ? ` · revoked ${formatDate(invite.revokedAt)}` : ''}
                </span>
              </li>
            ))}
          </ul>
        </details>
      )}

      <p className="ui-text-xs ui-text-muted">
        Revoking a code stops new students from joining with it. Students who
        already joined keep their access and their anonymous identity.
      </p>

      <ConfirmDialog
        open={confirming === 'revoke'}
        tone="danger"
        title="Revoke this class code?"
        description="Nobody new will be able to join with it. Students who already joined are not affected."
        confirmLabel="Revoke"
        busy={busy}
        onConfirm={() =>
          void run(() => (active ? revokeStudentInvite(courseId, active.invitationId) : Promise.resolve()))
        }
        onCancel={() => setConfirming(null)}
      />
      <ConfirmDialog
        open={confirming === 'replace'}
        title="Replace the class code?"
        description="The current code stops working and a new one is created. Update the board and any links you shared."
        confirmLabel="Create new code"
        busy={busy}
        onConfirm={() => void run(() => createStudentInvite(courseId, { replaceExisting: true }))}
        onCancel={() => setConfirming(null)}
      />
    </section>
  );
}
