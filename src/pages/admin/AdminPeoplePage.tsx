import { useCallback, useEffect, useMemo, useState } from 'react';
import { Button } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { ConfirmDialog } from '../../components/ui/ConfirmDialog';
import { FormField } from '../../components/ui/FormField';
import { PageHeader } from '../../components/ui/PageHeader';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { StatusPill } from '../../components/ui/StatusPill';
import { useSession } from '../../context/SessionContext';
import { useCourses } from '../../hooks/useCourses';
import {
  addMembership,
  createInvitation,
  createResetInvite,
  invitationLinkUrl,
  listInvitations,
  listUsers,
  removeMembership,
  revokeInvitation,
  updateUser,
  type AdminInvitation,
  type AdminUser,
  type CreatedInvitation,
} from '../../lib/adminPeopleApi';
import { formatCourseCode } from '../../lib/courseLabels';

type Loadable<T> =
  | { status: 'loading' }
  | { status: 'ready'; data: T }
  | { status: 'error'; message: string };

function errorText(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

function formatDate(iso: string | null | undefined): string {
  if (!iso) {
    return '—';
  }
  try {
    return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
  } catch {
    return iso;
  }
}

function origin(): string {
  return typeof window === 'undefined' ? '' : window.location.origin;
}

/**
 * The one-time link an invitation produces.
 *
 * Shown exactly once, in the response that created it; the backend stores
 * only a hash. Losing it means creating another, which is the correct cost.
 */
function CreatedLink({
  created,
  onDismiss,
}: {
  created: CreatedInvitation;
  onDismiss: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const link = invitationLinkUrl(origin(), created.path);

  async function copy() {
    try {
      await navigator.clipboard.writeText(link);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <Callout
      tone="success"
      title={
        created.kind === 'reset'
          ? 'Reset link created'
          : `${created.kind === 'admin' ? 'Administrator' : 'Instructor'} invitation created`
      }
      actions={
        <>
          <Button size="sm" variant="secondary" iconLeft="copy" onClick={() => void copy()}>
            {copied ? 'Copied' : 'Copy link'}
          </Button>
          <Button size="sm" variant="ghost" onClick={onDismiss}>
            Done
          </Button>
        </>
      }
    >
      <p>
        Send this link to the person yourself. It is shown only now, works once, and
        expires {formatDate(created.expiresAt)}.
      </p>
      <code className="people__link">{link}</code>
    </Callout>
  );
}

/**
 * People: accounts, roles, course assignments, and privileged invitations.
 *
 * Every control here maps to one administrator-only backend route, and the
 * backend refuses the same request from a professor session, so nothing on
 * this page is a permission — it is a convenient way to exercise one.
 */
export function AdminPeoplePage() {
  const { session } = useSession();
  const { state: coursesState } = useCourses();
  const [users, setUsers] = useState<Loadable<AdminUser[]>>({ status: 'loading' });
  const [invitations, setInvitations] = useState<Loadable<AdminInvitation[]>>({
    status: 'loading',
  });
  const [created, setCreated] = useState<CreatedInvitation | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [kind, setKind] = useState<'professor' | 'admin'>('professor');
  const [label, setLabel] = useState('');
  const [selectedCourses, setSelectedCourses] = useState<string[]>([]);
  const [assignments, setAssignments] = useState<Record<string, string>>({});
  const [pendingDisable, setPendingDisable] = useState<AdminUser | null>(null);

  const courses = useMemo(
    () => (coursesState.status === 'ready' ? coursesState.courses : []),
    [coursesState],
  );
  const courseName = useMemo(() => {
    const names = new Map<string, string>();
    for (const course of courses) {
      names.set(course.courseId, formatCourseCode(course.metadata.name) || course.courseId);
    }
    return (courseId: string) => names.get(courseId) ?? courseId;
  }, [courses]);

  const loadUsers = useCallback(async () => {
    try {
      setUsers({ status: 'ready', data: (await listUsers()).users });
    } catch (error) {
      setUsers({ status: 'error', message: errorText(error, 'Could not load accounts.') });
    }
  }, []);

  const loadInvitations = useCallback(async () => {
    try {
      setInvitations({ status: 'ready', data: (await listInvitations()).invitations });
    } catch (error) {
      setInvitations({
        status: 'error',
        message: errorText(error, 'Could not load invitations.'),
      });
    }
  }, []);

  useEffect(() => {
    void loadUsers();
    void loadInvitations();
  }, [loadUsers, loadInvitations]);

  async function run(action: () => Promise<unknown>, reload: 'users' | 'invitations' | 'both') {
    setBusy(true);
    setActionError(null);
    try {
      await action();
      if (reload !== 'invitations') {
        await loadUsers();
      }
      if (reload !== 'users') {
        await loadInvitations();
      }
    } catch (error) {
      setActionError(errorText(error, 'That did not work.'));
    } finally {
      setBusy(false);
    }
  }

  async function handleInvite(event: React.FormEvent) {
    event.preventDefault();
    await run(async () => {
      const result = await createInvitation({
        kind,
        ...(kind === 'professor' ? { courseIds: selectedCourses } : {}),
        ...(label.trim() ? { label: label.trim() } : {}),
      });
      setCreated(result);
      setLabel('');
      setSelectedCourses([]);
    }, 'invitations');
  }

  function toggleCourse(courseId: string) {
    setSelectedCourses((current) =>
      current.includes(courseId)
        ? current.filter((id) => id !== courseId)
        : [...current, courseId],
    );
  }

  const me = session.user?.userId;

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        eyebrow="Admin"
        title="People"
        description="Instructor and administrator accounts, their courses, and the invitations that create them."
      />

      {actionError && (
        <Callout tone="danger" title="That did not work">
          {actionError}
        </Callout>
      )}

      {created && <CreatedLink created={created} onDismiss={() => setCreated(null)} />}

      <section className="ui-stack ui-stack--snug">
        <SectionHeader
          title="Invite someone"
          description="Invitations work once and expire: instructors after 7 days, administrators after 24 hours."
          divider
        />
        <form className="people__invite" onSubmit={handleInvite} noValidate>
          <FormField label="Role" required>
            {({ id }) => (
              <select
                id={id}
                className="ui-input"
                value={kind}
                onChange={(event) => setKind(event.target.value as 'professor' | 'admin')}
                disabled={busy}
              >
                <option value="professor">Instructor</option>
                <option value="admin">Administrator</option>
              </select>
            )}
          </FormField>
          <FormField label="Note" optional hint="For your own records, e.g. the person's name.">
            {({ id }) => (
              <input
                id={id}
                className="ui-input"
                type="text"
                maxLength={120}
                value={label}
                onChange={(event) => setLabel(event.target.value)}
                disabled={busy}
              />
            )}
          </FormField>
          {kind === 'professor' && (
            <fieldset className="people__courses">
              <legend className="ui-field__label">Courses to assign</legend>
              {courses.length === 0 ? (
                <p className="ui-text-xs ui-text-muted">
                  No courses yet. The instructor can create one after signing in.
                </p>
              ) : (
                <ul className="chips" aria-label="Courses to assign">
                  {courses.map((course) => (
                    <li key={course.courseId}>
                      <label className="chip chip--choice">
                        <input
                          type="checkbox"
                          checked={selectedCourses.includes(course.courseId)}
                          onChange={() => toggleCourse(course.courseId)}
                          disabled={busy}
                        />
                        {courseName(course.courseId)}
                      </label>
                    </li>
                  ))}
                </ul>
              )}
            </fieldset>
          )}
          <div>
            <Button type="submit" variant="primary" iconLeft="add" loading={busy} loadingLabel="Creating…">
              Create invitation
            </Button>
          </div>
        </form>
      </section>

      <section className="ui-stack ui-stack--snug">
        <SectionHeader title="Accounts" divider />
        {users.status === 'loading' && (
          <p className="ui-text-muted" role="status" aria-live="polite">
            Loading accounts…
          </p>
        )}
        {users.status === 'error' && (
          <Callout tone="danger" title="Could not load accounts">
            {users.message}
          </Callout>
        )}
        {users.status === 'ready' && (
          <ul className="admin-rows people" aria-label="Accounts">
            {users.data.map((user) => {
              const assignable = courses.filter(
                (course) => !user.courseIds.includes(course.courseId),
              );
              const choice = assignments[user.userId] ?? '';
              return (
                <li key={user.userId} className="admin-row admin-row--stacked">
                  <div className="admin-row__main">
                    <p className="admin-row__label">
                      {user.displayName}{' '}
                      <span className="ui-text-muted">· {user.email}</span>{' '}
                      {user.userId === me && <span className="ui-text-muted">(you)</span>}
                    </p>
                    <p className="people__pills">
                      <StatusPill tone={user.role === 'admin' ? 'warning' : 'neutral'}>
                        {user.role === 'admin' ? 'administrator' : 'instructor'}
                      </StatusPill>{' '}
                      {user.disabled && <StatusPill tone="danger">disabled</StatusPill>}
                    </p>
                    <p className="ui-text-xs ui-text-muted">
                      Last sign-in {formatDate(user.lastLoginAt)}
                    </p>
                    {user.role === 'professor' && (
                      <div className="people__memberships">
                        <span className="ui-text-xs ui-text-muted">Courses:</span>
                        {user.courseIds.length === 0 && (
                          <span className="ui-text-xs ui-text-muted">none assigned</span>
                        )}
                        <ul className="chips" aria-label={`Courses for ${user.displayName}`}>
                          {user.courseIds.map((courseId) => (
                            <li key={courseId} className="chip">
                              {courseName(courseId)}
                              <button
                                type="button"
                                className="chip__remove"
                                aria-label={`Remove ${courseName(courseId)} from ${user.displayName}`}
                                disabled={busy}
                                onClick={() =>
                                  void run(() => removeMembership(user.userId, courseId), 'users')
                                }
                              >
                                ×
                              </button>
                            </li>
                          ))}
                        </ul>
                        {assignable.length > 0 && (
                          <span className="people__assign">
                            <label className="ui-visually-hidden" htmlFor={`assign-${user.userId}`}>
                              Assign a course to {user.displayName}
                            </label>
                            <select
                              id={`assign-${user.userId}`}
                              className="ui-input ui-input--sm"
                              value={choice}
                              disabled={busy}
                              onChange={(event) =>
                                setAssignments((current) => ({
                                  ...current,
                                  [user.userId]: event.target.value,
                                }))
                              }
                            >
                              <option value="">Assign a course…</option>
                              {assignable.map((course) => (
                                <option key={course.courseId} value={course.courseId}>
                                  {courseName(course.courseId)}
                                </option>
                              ))}
                            </select>
                            <Button
                              size="sm"
                              variant="secondary"
                              disabled={busy || !choice}
                              onClick={() =>
                                void run(async () => {
                                  await addMembership(user.userId, choice);
                                  setAssignments((current) => ({ ...current, [user.userId]: '' }));
                                }, 'users')
                              }
                            >
                              Add
                            </Button>
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                  <div className="admin-row__actions people__actions">
                    <Button
                      size="sm"
                      variant="secondary"
                      disabled={busy || user.disabled}
                      onClick={() =>
                        void run(async () => setCreated(await createResetInvite(user.userId)), 'invitations')
                      }
                    >
                      Reset link
                    </Button>
                    {user.userId !== me && (
                      <Button
                        size="sm"
                        variant="secondary"
                        disabled={busy}
                        onClick={() =>
                          void run(
                            () =>
                              updateUser(user.userId, {
                                role: user.role === 'admin' ? 'professor' : 'admin',
                              }),
                            'users',
                          )
                        }
                      >
                        {user.role === 'admin' ? 'Make instructor' : 'Make administrator'}
                      </Button>
                    )}
                    {user.userId !== me && (
                      <Button
                        size="sm"
                        variant={user.disabled ? 'secondary' : 'danger'}
                        disabled={busy}
                        onClick={() =>
                          user.disabled
                            ? void run(() => updateUser(user.userId, { disabled: false }), 'users')
                            : setPendingDisable(user)
                        }
                      >
                        {user.disabled ? 'Enable' : 'Disable'}
                      </Button>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      <section className="ui-stack ui-stack--snug">
        <SectionHeader
          title="Invitations"
          description="Instructor, administrator and reset links. Links themselves are never shown again."
          divider
        />
        {invitations.status === 'loading' && (
          <p className="ui-text-muted" role="status" aria-live="polite">
            Loading invitations…
          </p>
        )}
        {invitations.status === 'error' && (
          <Callout tone="danger" title="Could not load invitations">
            {invitations.message}
          </Callout>
        )}
        {invitations.status === 'ready' && invitations.data.length === 0 && (
          <p className="ui-text-muted">No invitations yet.</p>
        )}
        {invitations.status === 'ready' && invitations.data.length > 0 && (
          <ul className="admin-rows" aria-label="Invitations">
            {invitations.data.map((invitation) => (
              <li key={invitation.invitationId} className="admin-row">
                <span className="admin-row__label">
                  {invitation.kind === 'admin'
                    ? 'Administrator'
                    : invitation.kind === 'reset'
                      ? 'Password reset'
                      : 'Instructor'}
                  {invitation.label ? ` · ${invitation.label}` : ''}
                </span>
                <span className="admin-row__value">
                  {invitation.kind === 'professor' && invitation.courseIds.length > 0
                    ? `${invitation.courseIds.map(courseName).join(', ')} · `
                    : ''}
                  created {formatDate(invitation.createdAt)}
                  {invitation.createdByName ? ` by ${invitation.createdByName}` : ''}
                  {invitation.expiresAt ? ` · expires ${formatDate(invitation.expiresAt)}` : ''}
                </span>
                <StatusPill
                  tone={
                    invitation.status === 'active'
                      ? 'success'
                      : invitation.status === 'used'
                        ? 'neutral'
                        : 'danger'
                  }
                >
                  {invitation.status}
                </StatusPill>
                {invitation.status === 'active' && (
                  <Button
                    size="sm"
                    variant="ghost"
                    disabled={busy}
                    onClick={() =>
                      void run(() => revokeInvitation(invitation.invitationId), 'invitations')
                    }
                  >
                    Revoke
                  </Button>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>

      <ConfirmDialog
        open={pendingDisable !== null}
        tone="danger"
        title={`Disable ${pendingDisable?.displayName ?? 'this account'}?`}
        description="They are signed out everywhere and cannot sign in until the account is enabled again. Nothing they created is removed."
        confirmLabel="Disable"
        busy={busy}
        onConfirm={() => {
          const target = pendingDisable;
          setPendingDisable(null);
          if (target) {
            void run(() => updateUser(target.userId, { disabled: true }), 'users');
          }
        }}
        onCancel={() => setPendingDisable(null)}
      />
    </div>
  );
}
