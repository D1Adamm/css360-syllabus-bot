import { CourseRow } from '../../components/course/CourseRow';
import { LinkButton } from '../../components/ui/Button';
import { ErrorState } from '../../components/ui/ErrorState';
import { EmptyState } from '../../components/ui/EmptyState';
import { PageHeader } from '../../components/ui/PageHeader';
import { useCourses } from '../../hooks/useCourses';
import { toUserMessage } from '../../lib/errorMessages';
import { joinPath, studentCourseHomePath } from '../../lib/roleRoutes';

/**
 * The courses this browser remembers, and the way to join another.
 *
 * The backend lists only what this browser may open: every course it joined
 * with a class code, or every course a member of staff walking through the
 * student flow may reach. Returning to `/` with a single course skips this page
 * (see `RoleLanding`); arriving here on purpose always shows the list.
 */
export function StudentCoursesPage() {
  const { state, retry } = useCourses();

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        title="Your courses"
        description="Open a course to contribute questions, compare answers, and evaluate responses."
        actions={
          <LinkButton to={joinPath()} variant="secondary" iconLeft="add">
            Join another course
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
        <ul className="course-rows" aria-label="Your courses">
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
