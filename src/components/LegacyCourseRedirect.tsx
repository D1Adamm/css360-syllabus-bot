import { Navigate, useLocation } from 'react-router-dom';
import { defaultCoursePagePath, type CoursePageSegment } from '../lib/courseRoutes';

interface LegacyCourseRedirectProps {
  segment: CoursePageSegment;
}

/** Redirect legacy app paths to /course/css360-default/:segment, preserving query string. */
export function LegacyCourseRedirect({ segment }: LegacyCourseRedirectProps) {
  const location = useLocation();
  return <Navigate to={defaultCoursePagePath(segment, location.search)} replace />;
}
