import { useEffect, useId, useState } from 'react';
import { CourseExampleSummary } from '../../components/course/CourseExampleSummary';
import { CourseRow } from '../../components/course/CourseRow';
import { ErrorState } from '../../components/ui/ErrorState';
import { EmptyState } from '../../components/ui/EmptyState';
import { LinkButton } from '../../components/ui/Button';
import { PageHeader } from '../../components/ui/PageHeader';
import { StatusPill } from '../../components/ui/StatusPill';
import { useSession } from '../../context/SessionContext';
import { canStaffCourse, isAdmin, isParticipantOf } from '../../context/session';
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

/**
 * "My courses only", remembered per signed-in administrator in this browser.
 *
 * Keyed by user id so one person's choice never applies to another account
 * signed in on the same machine. Storage can be unavailable (private windows,
 * blocked site data); the page then simply starts unchecked.
 */
export function mineOnlyStorageKey(userId: string): string {
  return `sml.professorCourses.mineOnly.${userId}`;
}

function readMineOnly(userId: string | undefined): boolean {
  if (!userId) {
    return false;
  }
  try {
    return window.localStorage.getItem(mineOnlyStorageKey(userId)) === '1';
  } catch {
    return false;
  }
}

function writeMineOnly(userId: string, value: boolean): void {
  try {
    if (value) {
      window.localStorage.setItem(mineOnlyStorageKey(userId), '1');
    } else {
      window.localStorage.removeItem(mineOnlyStorageKey(userId));
    }
  } catch {
    // A preference, not data: nothing to report.
  }
}

export function ProfessorCoursesPage() {
  const { state, retry } = useCourses();
  const { session } = useSession();
  const checkboxId = useId();
  const userId = session.user?.userId;
  // Administrators reach every course; the list can be narrowed to the courses
  // they are explicitly assigned to teach. A filter on what is shown, never on
  // what they may open. Professors only ever receive their own courses.
  const admin = isAdmin(session);
  const [mineOnly, setMineOnly] = useState(() => readMineOnly(userId));
  useEffect(() => {
    setMineOnly(readMineOnly(userId));
  }, [userId]);
  const assigned = new Set(session.user?.courseIds ?? []);
  const narrowed = admin && mineOnly;
  // The list endpoint also returns a course this browser joined with a class
  // code. Its staff pages are refused, so it does not belong here. Everything
  // else the backend listed is kept even if the cached session has not caught
  // up with a new membership yet: the course guard re-checks on the way in.
  const courses =
    state.status === 'ready'
      ? state.courses.filter(
          ({ courseId }) =>
            (!isParticipantOf(session, courseId) || canStaffCourse(session, courseId)) &&
            (!narrowed || assigned.has(courseId)),
        )
      : [];

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        title="Courses"
        description={
          admin && !narrowed
            ? 'Every course in Syllabus Model Lab. You can open all of them as an administrator.'
            : 'Your courses in Syllabus Model Lab.'
        }
        actions={
          <LinkButton to="/professor/courses/new" variant="primary" iconLeft="add">
            Create course
          </LinkButton>
        }
      />

      {admin && (
        <div className="ui-checkbox-row">
          <input
            id={checkboxId}
            type="checkbox"
            checked={mineOnly}
            aria-describedby={`${checkboxId}-hint`}
            onChange={(event) => {
              setMineOnly(event.target.checked);
              if (userId) {
                writeMineOnly(userId, event.target.checked);
              }
            }}
          />
          <label htmlFor={checkboxId}>My courses only</label>
          <span id={`${checkboxId}-hint`} className="ui-text-xs ui-text-muted">
            Courses you are assigned to teach. Your administrator access is unchanged.
          </span>
        </div>
      )}

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

      {state.status === 'ready' && courses.length === 0 && narrowed && (
        <EmptyState
          illustration="empty-course"
          title="No courses are assigned to you"
          description="You are not assigned as an instructor on any course. Clear “My courses only” to see every course, or assign yourself to a course you teach under Admin → People."
        />
      )}

      {state.status === 'ready' && courses.length === 0 && !narrowed && (
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

      {state.status === 'ready' && courses.length > 0 && (
        <ul className="course-rows" aria-label="Your courses">
          {courses.map(({ courseId, metadata }) => {
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
