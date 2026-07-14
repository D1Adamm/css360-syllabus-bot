import { Link } from 'react-router-dom';
import { PageHeader } from './PageHeader';
import { DEFAULT_COURSE_ID } from '../lib/courseId';
import { coursePagePath } from '../lib/courseRoutes';

interface InvalidCoursePageProps {
  courseId?: string;
}

export function InvalidCoursePage({ courseId }: InvalidCoursePageProps) {
  const displayId = courseId && courseId.trim() !== '' ? courseId : '(missing)';
  const defaultHome = coursePagePath(DEFAULT_COURSE_ID, 'home');

  return (
    <section aria-labelledby="invalid-course-title">
      <PageHeader
        title="Invalid Course"
        description={`The course id "${displayId}" is not valid. Course ids must use lowercase letters, numbers, and hyphens only, and cannot begin or end with a hyphen.`}
      />
      <p>
        <Link to={defaultHome} className="button-link button-link--primary">
          Go to default course
        </Link>
      </p>
    </section>
  );
}
