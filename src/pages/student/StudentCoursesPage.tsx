import { Navigate } from 'react-router-dom';
import { CourseRow } from '../../components/course/CourseRow';
import { LinkButton } from '../../components/ui/Button';
import { ErrorState } from '../../components/ui/ErrorState';
import { EmptyState } from '../../components/ui/EmptyState';
import { PageHeader } from '../../components/ui/PageHeader';
import { useSession } from '../../context/SessionContext';
import { useCourses } from '../../hooks/useCourses';
import { toUserMessage } from '../../lib/errorMessages';
import { joinPath, studentCourseHomePath } from '../../lib/roleRoutes';

/**
 * Course selection for students.
 *
 * The backend lists only what this browser may open: the one course a
 * participant joined, or every course a member of staff walking through the
 * student flow may reach. A participant with exactly one course is sent
 * straight into it; a list of one is not a choice.
 */
export function StudentCoursesPage() {
  const { state, retry } = useCourses();
  const { session } = useSession();

  if (
    state.status === 'ready' &&
    state.courses.length === 1 &&
    session.participant &&
    !session.user
  ) {
    return <Navigate to={studentCourseHomePath(state.courses[0].courseId)} replace />;
  }

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        title="Your courses"
        description="Open a course to contribute questions, compare answers, and evaluate responses."
        actions={
          <LinkButton to={joinPath()} variant="secondary">
            Enter a class code
          </LinkButton>
        }
      />

      {state.status === 'loading' && (
        <p className="ui-text-muted" role="status" aria-live="polite">
          Loading your courses…
        </p>
      )}

      {state.status === 'error' && (
        <ErrorState
          title="Courses unavailable"
          message={
            toUserMessage(new Error(state.message), {
              audience: 'student',
              context: 'course-list',
            }).message
          }
          onRetry={retry}
        />
      )}

      {state.status === 'ready' && state.courses.length === 0 && (
        <EmptyState
          illustration="empty-course"
          size="full"
          title="No course yet"
          description="Enter the class code your instructor shared to join a course."
        />
      )}

      {state.status === 'ready' && state.courses.length > 0 && (
        <ul className="course-rows" aria-label="Available courses">
          {state.courses.map(({ courseId, metadata }) => (
            <CourseRow
              key={courseId}
              to={studentCourseHomePath(courseId)}
              name={metadata.name}
              title={metadata.title}
              meta={metadata.term}
            />
          ))}
        </ul>
      )}
    </div>
  );
}
