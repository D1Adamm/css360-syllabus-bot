import { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Button } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { FormField } from '../../components/ui/FormField';
import { PageHeader } from '../../components/ui/PageHeader';
import { useSession } from '../../context/SessionContext';
import { ApiError } from '../../lib/httpClient';
import { joinCourse } from '../../lib/authApi';
import { studentCourseHomePath } from '../../lib/roleRoutes';

/**
 * The student way in.
 *
 * Type the six-character code from the board, press Join, and land in the
 * course. A link that already carries the code (`/join/7K4P9X`, for Canvas)
 * submits itself. No name, no email, no account: what the backend creates is
 * an anonymous participant for this browser.
 *
 * The code is shown upper-case as it is typed, because that is how it is
 * written on the board and how the backend normalises it anyway.
 */
export function JoinPage() {
  const { code: codeFromLink } = useParams<{ code?: string }>();
  const { refresh } = useSession();
  const navigate = useNavigate();
  const [code, setCode] = useState(codeFromLink ?? '');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const autoSubmitted = useRef(false);

  async function submit(candidate: string) {
    const trimmed = candidate.trim();
    if (!trimmed) {
      setError('Enter the class code your instructor shared.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await joinCourse(trimmed);
      await refresh();
      navigate(studentCourseHomePath(result.courseId), { replace: true });
    } catch (caught) {
      setError(
        caught instanceof ApiError && caught.status === 429
          ? 'Too many attempts. Wait a minute and try again.'
          : caught instanceof Error
            ? caught.message
            : "That code isn't valid right now.",
      );
    } finally {
      setBusy(false);
    }
  }

  // A direct link joins on arrival, once. If it fails the code stays in the
  // field so the student can correct it.
  useEffect(() => {
    if (codeFromLink && !autoSubmitted.current) {
      autoSubmitted.current = true;
      void submit(codeFromLink);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [codeFromLink]);

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    void submit(code);
  }

  return (
    <div className="auth-page">
      <PageHeader
        title="Join your course"
        description="Enter the class code your instructor shared. No account or sign-in is needed."
      />

      <form className="auth-card" onSubmit={handleSubmit} noValidate>
        {error && (
          <Callout tone="danger" title="Couldn't join">
            {error}
          </Callout>
        )}
        <FormField label="Class code" hint="Six letters and numbers, for example 7K4P9X." required>
          {({ id, describedBy, invalid }) => (
            <input
              id={id}
              className="ui-input join-code-input"
              type="text"
              inputMode="text"
              autoComplete="off"
              autoCapitalize="characters"
              spellCheck={false}
              maxLength={12}
              value={code}
              onChange={(event) => setCode(event.target.value.toUpperCase())}
              aria-describedby={describedBy}
              aria-invalid={invalid}
              disabled={busy}
            />
          )}
        </FormField>
        <div className="auth-card__actions">
          <Button type="submit" variant="primary" loading={busy} loadingLabel="Joining…" iconRight="forward">
            Join
          </Button>
        </div>
        <p className="ui-text-xs ui-text-muted">
          Your name is never asked for and never stored. This browser is given an
          anonymous identity for the course; clearing your cookies starts a new one.
        </p>
      </form>
    </div>
  );
}
