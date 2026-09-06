import { useLocation } from 'react-router-dom';
import { LinkButton } from '../components/ui/Button';
import { EmptyState } from '../components/ui/EmptyState';
import { PageHeader } from '../components/ui/PageHeader';
import { useSession } from '../context/SessionContext';
import { isStaff, roleForSession } from '../context/session';
import { joinPath, roleHomePath } from '../lib/roleRoutes';

/**
 * Signed in, but not here.
 *
 * A professor opening another instructor's course, a student opening a
 * professor page, an instructor opening administration. The page says which
 * it was in plain words and offers the way back; it never suggests the thing
 * exists behind a different door.
 */
export function ForbiddenPage() {
  const { session } = useSession();
  const location = useLocation();
  const from = (location.state as { from?: string } | null)?.from;
  const staff = isStaff(session);

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader title="You don't have access to that" />
      <EmptyState
        size="full"
        illustration="empty-course"
        title={staff ? 'This page is outside your courses' : 'This course is not the one you joined'}
        description={
          staff
            ? 'Professors see the courses an administrator has assigned to them, and administration is limited to administrators. If you think this is a mistake, ask an administrator to check your assignments.'
            : 'Each class code opens one course. To use another course, enter the code your instructor shared for it.'
        }
        action={
          staff ? (
            <LinkButton to={roleHomePath(roleForSession(session))} variant="primary">
              Back to your courses
            </LinkButton>
          ) : (
            <LinkButton to={joinPath()} variant="primary">
              Enter a class code
            </LinkButton>
          )
        }
      />
      {from && (
        <p className="ui-text-xs ui-text-muted">
          You tried to open <code>{from}</code>.
        </p>
      )}
    </div>
  );
}
