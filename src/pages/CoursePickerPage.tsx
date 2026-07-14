import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { PageHeader } from '../components/PageHeader';
import {
  subscribeToCourses,
  type CourseListItem,
} from '../lib/coursesDb';
import { coursePagePath } from '../lib/courseRoutes';
import { formatSyllabusStatusLabel } from '../lib/syllabusStatusLabel';

type CoursePickerState =
  | { status: 'loading' }
  | { status: 'ready'; courses: CourseListItem[] }
  | { status: 'error'; message: string };

function statusClassName(label: string): string {
  switch (label) {
    case 'Indexed':
      return 'course-status-label course-status-label--indexed';
    case 'Extracted':
      return 'course-status-label course-status-label--extracted';
    case 'Uploaded':
      return 'course-status-label course-status-label--uploaded';
    case 'Failed':
      return 'course-status-label course-status-label--failed';
    default:
      return 'course-status-label course-status-label--none';
  }
}

export function CoursePickerPage() {
  const [state, setState] = useState<CoursePickerState>({ status: 'loading' });

  useEffect(() => {
    setState({ status: 'loading' });

    const unsubscribe = subscribeToCourses(
      (courses) => {
        setState({ status: 'ready', courses });
      },
      (message) => {
        setState({
          status: 'error',
          message: message || 'Could not load courses from Firebase.',
        });
      },
    );

    return unsubscribe;
  }, []);

  return (
    <>
      <PageHeader
        title="Courses"
        description="Choose a course to open its syllabus lab, or create a new course."
      />

      <div className="course-picker-actions">
        <Link to="/create-course" className="button-link button-link--primary">
          Create Course
        </Link>
      </div>

      {state.status === 'loading' ? (
        <section className="course-picker-state" aria-live="polite" aria-busy="true">
          <h2 className="course-picker-state__title">Loading courses</h2>
          <p className="course-picker-state__text">
            Fetching available courses from Firebase…
          </p>
        </section>
      ) : null}

      {state.status === 'error' ? (
        <section className="course-picker-state" aria-live="polite">
          <h2 className="course-picker-state__title">Firebase error</h2>
          <p className="course-picker-state__text">{state.message}</p>
        </section>
      ) : null}

      {state.status === 'ready' && state.courses.length === 0 ? (
        <section className="course-picker-state" aria-live="polite">
          <h2 className="course-picker-state__title">No courses available</h2>
          <p className="course-picker-state__text">
            No courses have been created yet. Use Create Course to add the first one.
          </p>
        </section>
      ) : null}

      {state.status === 'ready' && state.courses.length > 0 ? (
        <section className="course-picker-list" aria-label="Available courses">
          <ul className="course-picker-list__items">
            {state.courses.map(({ courseId, metadata }) => {
              const statusLabel = formatSyllabusStatusLabel(metadata.syllabusStatus);
              const instructorName = metadata.instructorName.trim();

              return (
                <li key={courseId} className="course-picker-card">
                  <div className="course-picker-card__header">
                    <h2 className="course-picker-card__name">{metadata.name}</h2>
                    <span className={statusClassName(statusLabel)}>{statusLabel}</span>
                  </div>
                  <p className="course-picker-card__title">{metadata.title}</p>
                  <dl className="course-picker-card__meta">
                    <div>
                      <dt>Term</dt>
                      <dd>{metadata.term}</dd>
                    </div>
                    {instructorName ? (
                      <div>
                        <dt>Instructor</dt>
                        <dd>{instructorName}</dd>
                      </div>
                    ) : null}
                    {metadata.chunkCount > 0 ? (
                      <div>
                        <dt>Chunks</dt>
                        <dd>{metadata.chunkCount}</dd>
                      </div>
                    ) : null}
                  </dl>
                  <p className="course-picker-card__id">
                    <span className="course-picker-card__id-label">Course id</span>{' '}
                    <code>{courseId}</code>
                  </p>
                  <Link
                    to={coursePagePath(courseId, 'home')}
                    className="button-link button-link--secondary"
                  >
                    Open Course
                  </Link>
                </li>
              );
            })}
          </ul>
        </section>
      ) : null}
    </>
  );
}
