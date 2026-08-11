import { CourseExampleSummary } from '../../components/course/CourseExampleSummary';
import { CourseRow } from '../../components/course/CourseRow';
import { ErrorState } from '../../components/ui/ErrorState';
import { EmptyState } from '../../components/ui/EmptyState';
import { LinkButton } from '../../components/ui/Button';
import { PageHeader } from '../../components/ui/PageHeader';
import { StatusPill } from '../../components/ui/StatusPill';
import { useCourses } from '../../hooks/useCourses';
import { toUserMessage } from '../../lib/errorMessages';
import { professorCourseHomePath } from '../../lib/roleRoutes';
import type { SyllabusStatus } from '../../types';

/**
 * Professor course list.
 *
 * Deliberately omits course ids and chunk counts — those are index internals,
 * and a professor has no decision to make with them. They remain visible in
 * the admin course list.
 */

interface SyllabusPresentation {
  label: string;
  tone: 'neutral' | 'success' | 'warning' | 'danger' | 'progress';
}

function syllabusPresentation(status: SyllabusStatus): SyllabusPresentation {
  switch (status) {
    case 'indexed':
    case 'ready':
      return { label: 'Syllabus ready', tone: 'success' };
    case 'processing':
      return { label: 'Preparing syllabus', tone: 'progress' };
    case 'uploaded':
    case 'extracted':
      return { label: 'Preparing syllabus', tone: 'progress' };
    case 'upload_failed':
    case 'index_failed':
    case 'error':
      return { label: 'Needs attention', tone: 'danger' };
    default:
      return { label: 'No syllabus yet', tone: 'neutral' };
  }
}

export function ProfessorCoursesPage() {
  const { state, retry } = useCourses();

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        title="Courses"
        description="Your courses in Syllabus Model Lab."
        actions={
          <LinkButton to="/professor/courses/new" variant="primary" iconLeft="add">
            Create course
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
              audience: 'professor',
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
          title="No courses yet"
          description="Create your first course and upload its syllabus to get started."
          action={
            <LinkButton to="/professor/courses/new" variant="primary">
              Create course
            </LinkButton>
          }
        />
      )}

      {state.status === 'ready' && state.courses.length > 0 && (
        <ul className="course-rows" aria-label="Your courses">
          {state.courses.map(({ courseId, metadata }) => {
            const syllabus = syllabusPresentation(metadata.syllabusStatus);
            const instructor = metadata.instructorName.trim();

            return (
              <CourseRow
                key={courseId}
                to={professorCourseHomePath(courseId)}
                name={metadata.name}
                title={metadata.title}
                meta={[metadata.term, instructor].filter(Boolean).join(' · ')}
                detail={<CourseExampleSummary courseId={courseId} />}
                status={<StatusPill tone={syllabus.tone}>{syllabus.label}</StatusPill>}
              />
            );
          })}
        </ul>
      )}
    </div>
  );
}
