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
    return <InvalidCoursePage />;
  }

  return (
    /*
     * `key` is load-bearing, not decoration.
     *
     * Every course page sits under this one route pattern, so React Router
     * reuses the same component instances when only `:courseId` changes — which
     * meant a comparison run, its four answers, and the active question all
     * survived a switch from one course to another. Keying on the course id
     * forces a remount, so no course-scoped state can outlive the course it
     * belongs to. Fixing it here covers every current and future course page
     * rather than asking each one to remember to reset itself.
     */
    <CourseProvider key={courseId} courseId={courseId}>
      <Outlet />
    </CourseProvider>
  );
}

/** Redirect /course/:courseId → /course/:courseId/home (relative). */
export function CourseIndexRedirect() {
  return <Navigate to="home" replace />;
}
