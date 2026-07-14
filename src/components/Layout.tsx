import { Link, Outlet, useLocation } from 'react-router-dom';
import { getCourseIdFromPathname } from '../lib/courseRoutes';
import { Navigation } from './Navigation';
import { PrototypeBanner } from './PrototypeBanner';

export function Layout() {
  const { pathname } = useLocation();
  const courseId = getCourseIdFromPathname(pathname);

  return (
    <div className="app-layout">
      <PrototypeBanner />
      <header className="app-header">
        <div className="app-header__inner">
          <Link to="/" className="app-logo">
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
