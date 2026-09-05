import { useState } from 'react';
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { Button } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { FormField } from '../../components/ui/FormField';
import { PageHeader } from '../../components/ui/PageHeader';
import { useSession } from '../../context/SessionContext';
import { isStaff, roleForSession } from '../../context/session';
import { ApiError } from '../../lib/httpClient';
import { login } from '../../lib/authApi';
import { joinPath, roleHomePath } from '../../lib/roleRoutes';

/**
 * Sign-in for professors and administrators.
 *
 * Students never see a password: their way in is the class code on `/join`,
 * and the page says so before anyone types an email address into the wrong
 * form. A wrong password and an unknown address get the same message, because
 * the backend deliberately does not distinguish them.
 */
export function LoginPage() {
  const { state, session, refresh } = useSession();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const from = (location.state as { from?: string } | null)?.from;

  if (state.status === 'ready' && isStaff(session)) {
    return <Navigate to={from ?? roleHomePath(roleForSession(session))} replace />;
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!email.trim() || !password) {
      setError('Enter your email address and password.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await login(email.trim(), password);
      await refresh();
      navigate(from ?? '/', { replace: true });
    } catch (caught) {
      setError(
        caught instanceof ApiError && caught.status === 429
          ? 'Too many attempts. Wait a little and try again.'
          : caught instanceof Error
            ? caught.message
            : 'Could not sign in.',
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-page">
      <PageHeader
        title="Sign in"
        description="For instructors and administrators. Students use their class code instead."
      />

      <div className="auth-columns">
        <form className="auth-card" onSubmit={handleSubmit} noValidate>
          {error && (
            <Callout tone="danger" title="Not signed in">
              {error}
            </Callout>
          )}
          <FormField label="Email address" required>
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
          <FormField label="Password" required>
            {({ id, describedBy, invalid }) => (
              <input
                id={id}
                className="ui-input"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                aria-describedby={describedBy}
                aria-invalid={invalid}
                disabled={busy}
              />
            )}
          </FormField>
          <div className="auth-card__actions">
            <Button type="submit" variant="primary" loading={busy} loadingLabel="Signing in…">
              Sign in
            </Button>
          </div>
          <p className="ui-text-xs ui-text-muted">
            Forgotten your password? Ask an administrator for a reset link.
          </p>
        </form>

        <aside className="auth-card auth-card--aside">
          <h2 className="auth-card__title">Students</h2>
          <p>
            You don&apos;t need an account. Enter the class code your instructor
            shared and you&apos;re in.
          </p>
          <Link to={joinPath()} className="ui-button ui-button--secondary ui-button--md">
            Enter a class code
          </Link>
        </aside>
      </div>
    </div>
  );
}
