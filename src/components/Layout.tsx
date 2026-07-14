import { Link, Outlet, useLocation } from 'react-router-dom';
import { DEFAULT_COURSE_ID } from '../lib/courseId';
import { coursePagePath, getCourseIdFromPathname } from '../lib/courseRoutes';
import { Navigation } from './Navigation';
import { PrototypeBanner } from './PrototypeBanner';

function useNavCourseId(): string {
  const { pathname } = useLocation();
  return getCourseIdFromPathname(pathname) ?? DEFAULT_COURSE_ID;
}

export function Layout() {
  const courseId = useNavCourseId();
  const homePath = coursePagePath(courseId, 'home');

  return (
    <div className="app-layout">
      <PrototypeBanner />
      <header className="app-header">
        <div className="app-header__inner">
          <Link to={homePath} className="app-logo">
            Syllabus Model Lab
          </Link>
          <Navigation courseId={courseId} />
        </div>
      </header>
      <main className="app-main">
        <Outlet />
      </main>
      <footer className="app-footer">
        <div className="app-footer__inner">
          <p>Syllabus Model Lab — Classroom prototype for model comparison research.</p>
        </div>
      </footer>
    </div>
  );
}
