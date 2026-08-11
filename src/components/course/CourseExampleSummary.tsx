import { useCourseExampleCounts } from '../../hooks/useCourseExampleCounts';
import { StatusPill } from '../ui/StatusPill';

export interface CourseExampleSummaryProps {
  courseId: string;
  /** Show the approved total as well as anything pending. */
  showApproved?: boolean;
}

/**
 * Review counts for a course, drawn from the examples themselves.
 *
 * Never derived from the approved-export status — an export is a training
 * artefact that can be stale or absent, and "48 approved" has to mean the
 * review state right now.
 *
 * Renders nothing while loading or unavailable rather than a zero, so an older
 * course with no examples recorded does not read as "0 approved".
 */
export function CourseExampleSummary({
  courseId,
  showApproved = true,
}: CourseExampleSummaryProps) {
  const state = useCourseExampleCounts(courseId);

  if (state.status !== 'ready') {
    return null;
  }

  const { approved, pending, total } = state.counts;

  if (total === 0) {
    return <span className="course-row__count">No examples yet</span>;
  }

  return (
    <span className="course-row__counts">
      {pending > 0 && (
        <StatusPill tone="warning">
          {pending} awaiting review
        </StatusPill>
      )}
      {showApproved && (
        <span className="course-row__count">{approved} approved</span>
      )}
    </span>
  );
}
