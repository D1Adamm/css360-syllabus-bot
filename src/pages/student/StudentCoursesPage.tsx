import { CourseRow } from '../../components/course/CourseRow';
import { ErrorState } from '../../components/ui/ErrorState';
import { EmptyState } from '../../components/ui/EmptyState';
import { PageHeader } from '../../components/ui/PageHeader';
import { useCourses } from '../../hooks/useCourses';
import { toUserMessage } from '../../lib/errorMessages';
import { studentCourseHomePath } from '../../lib/roleRoutes';

/**
 * Course selection for students.
 *
 * There is no enrolment backend, so this lists the courses that exist rather
 * than pretending to know which ones a student belongs to. Joining a course
 * with a code is a separate, later piece of work with real backend support;
 * nothing here simulates it.
 */
export function StudentCoursesPage() {
  const { state, retry } = useCourses();

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        title="Your courses"
        description="Open a course to contribute questions, compare answers, and evaluate responses."
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
          title="No courses available yet"
          description="When your instructor sets up a course here, it will appear on this page."
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
