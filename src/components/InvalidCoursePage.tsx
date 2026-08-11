import { EmptyState } from './ui/EmptyState';
import { LinkButton } from './ui/Button';

/**
 * Shown when a course URL contains an id this application cannot use.
 *
 * The id format is an internal detail, so it is no longer explained here — a
 * bad link is not something the reader can fix by learning our naming rules.
 */
export function InvalidCoursePage() {
  return (
    <EmptyState
      size="full"
      illustration="empty-course"
      title="We couldn't find that course"
      description="The link you followed doesn't point to a course in Syllabus Model Lab. It may have been mistyped or the course may have been removed."
      action={
        <LinkButton to="/" variant="primary">
          Go to your courses
        </LinkButton>
      }
    />
  );
}
