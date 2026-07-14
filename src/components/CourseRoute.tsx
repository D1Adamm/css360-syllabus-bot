import { Navigate, Outlet, useParams } from 'react-router-dom';
import { CourseProvider } from '../context/CourseContext';
import { isValidCourseId } from '../lib/courseId';
import { InvalidCoursePage } from './InvalidCoursePage';

/**
 * Validates :courseId from the URL, exposes it via CourseProvider, and renders
 * nested course page routes.
 */
export function CourseRoute() {
  const { courseId } = useParams<{ courseId: string }>();

  if (!isValidCourseId(courseId)) {
    return <InvalidCoursePage courseId={courseId} />;
  }

  return (
    <CourseProvider courseId={courseId}>
      <Outlet />
    </CourseProvider>
  );
}

/** Redirect /course/:courseId → /course/:courseId/home (relative). */
export function CourseIndexRedirect() {
  return <Navigate to="home" replace />;
}
