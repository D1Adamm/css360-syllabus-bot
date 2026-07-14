import { Link } from 'react-router-dom';
import { PageHeader } from './PageHeader';

interface InvalidCoursePageProps {
  courseId?: string;
}

export function InvalidCoursePage({ courseId }: InvalidCoursePageProps) {
  const displayId = courseId && courseId.trim() !== '' ? courseId : '(missing)';

  return (
    <section aria-labelledby="invalid-course-title">
      <PageHeader
        title="Invalid Course"
        description={`The course id "${displayId}" is not valid. Course ids must use lowercase letters, numbers, and hyphens only, and cannot begin or end with a hyphen.`}
      />
      <p>
        <Link to="/" className="button-link button-link--primary">
          Back to Courses
        </Link>
      </p>
    </section>
  );
}
