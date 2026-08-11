import { Outlet, useLocation } from 'react-router-dom';
import { useRole } from '../context/RoleContext';
import { useCourseMetadata } from '../hooks/useCourseMetadata';
import { getCourseIdFromPathname } from '../lib/courseRoutes';
import { getRoleAreaFromPathname, roleHomePath } from '../lib/roleRoutes';
import { BrandMark } from './BrandMark';
import { CourseContextBar } from './CourseContextBar';
import { DevRoleSwitcher } from './DevRoleSwitcher';
import { courseNavItems, primaryNavItems } from './navigation';
import { PrimaryNav } from './PrimaryNav';
import './shell.css';

/**
 * The single application shell.
 *
 * Which role's chrome renders is decided by the URL, not by the remembered
 * development role. A deep link into a professor page shows professor
 * navigation even if the switcher was last left on Student, so the URL and the
 * chrome can never disagree.
 */
export function AppShell() {
  const { pathname } = useLocation();
  const { role: storedRole } = useRole();

  const area = getRoleAreaFromPathname(pathname) ?? storedRole;
  const courseId = getCourseIdFromPathname(pathname);
  const { metadata } = useCourseMetadata(courseId);

  const navItems = primaryNavItems(area, courseId);
  const subNavItems = courseId ? courseNavItems(area, courseId) : [];
  const isAdmin = area === 'admin';

  return (
    <div className={`ui-root shell shell--${area}`}>
      <a className="ui-skip-link" href="#main-content">
        Skip to content
      </a>

      <header className="shell-header">
        <div className="ui-container shell-header__inner">
          <BrandMark to={roleHomePath(area)} />
          {!isAdmin && <PrimaryNav items={navItems} />}
          <div className="shell-header__end">
            <DevRoleSwitcher value={area} />
          </div>
        </div>
      </header>

      {courseId && !isAdmin && (
        <CourseContextBar
          name={metadata?.name || 'Course'}
          title={metadata?.title}
          term={metadata?.term}
          items={subNavItems}
        />
      )}

      <div className={isAdmin ? 'shell-body shell-body--admin' : 'shell-body'}>
        {isAdmin && (
          <aside className="shell-sidebar" aria-label="Admin sections">
            <PrimaryNav items={navItems} label="Admin navigation" collapsible={false} />
          </aside>
        )}

        <main id="main-content" className="shell-main">
          <div className="ui-container shell-main__inner">
            <Outlet />
          </div>
        </main>
      </div>

      <footer className="shell-footer">
        <div className="ui-container shell-footer__inner">
          <p>
            Syllabus Model Lab — a teaching and research project at UW Bothell.
            Not an official University of Washington service.
          </p>
        </div>
      </footer>
    </div>
  );
}
