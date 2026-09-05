import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Button } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { EmptyState } from '../../components/ui/EmptyState';
import { FormField } from '../../components/ui/FormField';
import { PageHeader } from '../../components/ui/PageHeader';
import { useSession } from '../../context/SessionContext';
import { roleForSession } from '../../context/session';
import {
  acceptInvitation,
  previewInvitation,
  type InvitationPreview,
} from '../../lib/authApi';
import { formatCourseCode } from '../../lib/courseLabels';
import { loginPath, roleHomePath } from '../../lib/roleRoutes';

const MIN_PASSWORD_LENGTH = 12;

type PreviewState =
  | { status: 'loading' }
  | { status: 'ready'; preview: InvitationPreview }
  | { status: 'invalid'; message: string };

const KIND_TITLE: Record<InvitationPreview['kind'], string> = {
  professor: 'Create your instructor account',
  admin: 'Create your administrator account',
  reset: 'Choose a new password',
};

/**
 * Accepting a professor, administrator, or password-reset invitation.
 *
 * The link is the credential: whoever opens it within its lifetime becomes
 * the account it describes. The page therefore shows exactly what accepting
 * will do — the role and the courses — before asking for anything, and the
 * backend consumes the invitation in the same transaction that creates the
 * account, so a used link cannot be used twice.
 */
export function AcceptInvitePage() {
  const { token = '' } = useParams<{ token: string }>();
  const { setSession } = useSession();
  const navigate = useNavigate();
  const [preview, setPreview] = useState<PreviewState>({ status: 'loading' });
  const [email, setEmail] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setPreview({ status: 'loading' });
    previewInvitation(token)
      .then((result) => {
        if (!cancelled) {
          setPreview({ status: 'ready', preview: result });
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          setPreview({
            status: 'invalid',
            message:
              caught instanceof Error ? caught.message : 'This invitation link is not valid.',
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [token]);

  if (preview.status === 'loading') {
    return (
      <div className="auth-page">
        <PageHeader title="Invitation" />
        <p className="ui-text-muted" role="status" aria-live="polite">
          Checking the invitation…
        </p>
      </div>
    );
  }

  if (preview.status === 'invalid') {
    return (
      <div className="auth-page">
        <PageHeader title="Invitation" />
        <EmptyState
          size="full"
          illustration="empty-course"
          title="This invitation can't be used"
          description={`${preview.message} Invitations work once and expire; ask the administrator who sent it for a new one.`}
        />
      </div>
    );
  }

  const { kind, courses, targetEmail } = preview.preview;
  const isReset = kind === 'reset';

  function validate(): Record<string, string> {
    const next: Record<string, string> = {};
    if (!isReset) {
      if (!email.trim() || !email.includes('@')) {
        next.email = 'Enter the email address you will sign in with.';
      }
      if (!displayName.trim()) {
        next.displayName = 'Enter the name to show to others.';
      }
    }
    if (password.length < MIN_PASSWORD_LENGTH) {
      next.password = `Use at least ${MIN_PASSWORD_LENGTH} characters.`;
    } else if (password !== confirm) {
      next.confirm = 'The two passwords do not match.';
    }
    return next;
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    const next = validate();
    setErrors(next);
    if (Object.keys(next).length > 0) {
      return;
    }
    setBusy(true);
    setSubmitError(null);
    try {
      const session = await acceptInvitation(token, {
        password,
        ...(isReset ? {} : { email: email.trim(), displayName: displayName.trim() }),
      });
      setSession(session);
      navigate(roleHomePath(roleForSession(session)), { replace: true });
    } catch (caught) {
      setSubmitError(
        caught instanceof Error ? caught.message : 'The invitation could not be accepted.',
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-page">
      <PageHeader
        title={KIND_TITLE[kind]}
        description={
          isReset
            ? `For the account ${targetEmail ?? ''}. Every other session for it will be signed out.`
            : kind === 'admin'
              ? 'This link creates an administrator account with access to every course and to administration.'
              : courses.length > 0
                ? `This link creates an instructor account for ${courses
                    .map((course) => formatCourseCode(course.name) || course.courseId)
                    .join(', ')}.`
                : 'This link creates an instructor account. An administrator will assign your courses.'
        }
      />

      <form className="auth-card" onSubmit={handleSubmit} noValidate>
        {submitError && (
          <Callout
            tone="danger"
            title="Not accepted"
            actions={
              submitError.toLowerCase().includes('already exists') ? (
                <Button variant="secondary" size="sm" onClick={() => navigate(loginPath())}>
                  Sign in instead
                </Button>
              ) : undefined
            }
          >
            {submitError}
          </Callout>
        )}

        {!isReset && (
          <>
            <FormField label="Email address" required error={errors.email}>
              {({ id, describedBy, invalid }) => (
                <input
                  id={id}
                  className="ui-input"
                  type="email"
                  autoComplete="username"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  aria-describedby={describedBy}
                  aria-invalid={invalid}
                  disabled={busy}
                />
              )}
            </FormField>
            <FormField label="Your name" hint="Shown to other staff." required error={errors.displayName}>
              {({ id, describedBy, invalid }) => (
                <input
                  id={id}
                  className="ui-input"
                  type="text"
                  autoComplete="name"
                  value={displayName}
                  onChange={(event) => setDisplayName(event.target.value)}
                  aria-describedby={describedBy}
                  aria-invalid={invalid}
                  disabled={busy}
                />
              )}
            </FormField>
          </>
        )}

        <FormField
          label="Password"
          hint={`At least ${MIN_PASSWORD_LENGTH} characters. A phrase is easier to remember than a code.`}
          required
          error={errors.password}
        >
          {({ id, describedBy, invalid }) => (
            <input
              id={id}
              className="ui-input"
              type="password"
              autoComplete="new-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              aria-describedby={describedBy}
              aria-invalid={invalid}
              disabled={busy}
            />
          )}
        </FormField>
        <FormField label="Confirm password" required error={errors.confirm}>
          {({ id, describedBy, invalid }) => (
            <input
              id={id}
              className="ui-input"
              type="password"
              autoComplete="new-password"
              value={confirm}
              onChange={(event) => setConfirm(event.target.value)}
              aria-describedby={describedBy}
              aria-invalid={invalid}
              disabled={busy}
            />
          )}
        </FormField>

        <div className="auth-card__actions">
          <Button type="submit" variant="primary" loading={busy} loadingLabel="Saving…">
            {isReset ? 'Set password and sign in' : 'Create account and sign in'}
          </Button>
        </div>
      </form>
    </div>
  );
}
