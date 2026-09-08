import { Navigate, Outlet, useParams } from 'react-router-dom';
import { AdminPreviewProvider } from '../context/AdminPreviewContext';
import { CourseProvider } from '../context/CourseContext';
import { isValidCourseId } from '../lib/courseId';
import { InvalidCoursePage } from './InvalidCoursePage';

export interface CourseRouteProps {
  /**
   * Set by the student tree only. An administrator who opens these pages
   * without having joined the course is previewing them, and the pages below
   * need to know so that nothing they submit is saved. See
   * `AdminPreviewContext`. The professor and admin trees leave it unset.
   */
  adminPreview?: boolean;
}

/**
 * Validates :courseId from the URL, exposes it via CourseProvider, and renders
 * nested course page routes.
 */
export function CourseRoute({ adminPreview = false }: CourseRouteProps = {}) {
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
      {adminPreview ? (
        <AdminPreviewProvider courseId={courseId}>
          <Outlet />
        </AdminPreviewProvider>
      ) : (
        <Outlet />
      )}
    </CourseProvider>
  );
}

/** Redirect /course/:courseId → /course/:courseId/home (relative). */
export function CourseIndexRedirect() {
  return <Navigate to="home" replace />;
}
