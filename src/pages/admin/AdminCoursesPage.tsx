import { CourseRow } from '../../components/course/CourseRow';
import { ErrorState } from '../../components/ui/ErrorState';
import { PageHeader } from '../../components/ui/PageHeader';
import { StatusPill } from '../../components/ui/StatusPill';
import { useCourses } from '../../hooks/useCourses';
import { formatSyllabusStatusLabel } from '../../lib/syllabusStatusLabel';
import { adminCoursePath } from '../../lib/roleRoutes';

/** Technical course listing. Course ids, index state and chunk counts belong here. */
export function AdminCoursesPage() {
  const { state, retry } = useCourses();

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        title="Courses"
        eyebrow="Admin"
        description="Course records, syllabus index state, and per-course diagnostics."
      />

      {state.status === 'loading' && (
        <p className="ui-text-muted" role="status" aria-live="polite">
          Loading courses…
        </p>
      )}

      {state.status === 'error' && (
        <ErrorState
          title="Could not read courses"
          message="The course list could not be loaded from the database."
          technical={state.message}
          onRetry={retry}
        />
      )}

      {state.status === 'ready' && (
        <ul className="course-rows" aria-label="Courses">
          {state.courses.map(({ courseId, metadata }) => (
            <CourseRow
              key={courseId}
              to={adminCoursePath(courseId)}
              name={metadata.name}
              title={metadata.title}
              meta={courseId}
              detail={`${metadata.term} · chunks: ${metadata.chunkCount} · file: ${
                metadata.syllabusFileName || '—'
              }`}
              status={
                <StatusPill
                  tone={
                    metadata.syllabusStatus === 'indexed' ||
                    metadata.syllabusStatus === 'ready'
                      ? 'success'
                      : metadata.syllabusStatus.includes('failed') ||
                          metadata.syllabusStatus === 'error'
                        ? 'danger'
                        : 'neutral'
                  }
                >
                  {formatSyllabusStatusLabel(metadata.syllabusStatus)}
                </StatusPill>
              }
            />
          ))}
        </ul>
      )}
    </div>
  );
}
