import { Link, Outlet, useLocation, useNavigate } from 'react-router-dom';
import { Button } from '../components/ui/Button';
import { useSession } from '../context/SessionContext';
import { isAnonymous, roleForSession } from '../context/session';
import { useCourseMetadata } from '../hooks/useCourseMetadata';
import { getCourseIdFromPathname } from '../lib/courseRoutes';
import { getRoleAreaFromPathname, loginPath, roleHomePath } from '../lib/roleRoutes';
import { BrandMark } from './BrandMark';
import { CourseContextBar } from './CourseContextBar';
import { courseNavItems, primaryNavItems } from './navigation';
import { PrimaryNav } from './PrimaryNav';
import './shell.css';

/**
 * The single application shell.
 *
 * Which role's chrome renders is decided by the URL first and the session
 * second: a professor deep link shows professor navigation, and a
 * role-neutral page such as sign-in draws the chrome for whoever is signed
 * in. Nothing about the chrome is a permission — the route guards decide what
 * renders inside it, and the backend decides what any of it can fetch.
 */
export function AppShell() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const { session, signOut } = useSession();

  const area = getRoleAreaFromPathname(pathname) ?? roleForSession(session);
  const courseId = getCourseIdFromPathname(pathname);
  const { state: metadataState, metadata } = useCourseMetadata(courseId);

  const anonymous = isAnonymous(session);
  const navItems = anonymous ? [] : primaryNavItems(area, courseId);
  const subNavItems = courseId && !anonymous ? courseNavItems(area, courseId) : [];
  const isAdmin = area === 'admin';

  async function handleSignOut() {
    await signOut();
    navigate(loginPath(), { replace: true });
  }

  return (
    <div className={`ui-root shell shell--${area}`}>
      <a className="ui-skip-link" href="#main-content">
        Skip to content
      </a>

      <header className="shell-header">
        <div className="ui-container shell-header__inner">
          <BrandMark to={anonymous ? '/' : roleHomePath(area)} />
          {!isAdmin && <PrimaryNav items={navItems} />}
          <div className="shell-header__end">
            {session.user ? (
              <div className="shell-identity">
                <span className="shell-identity__name" title={session.user.email}>
                  {session.user.displayName}
                </span>
                <span className="shell-identity__role">
                  {session.user.role === 'admin' ? 'Administrator' : 'Instructor'}
                </span>
                <Button size="sm" variant="ghost" onClick={() => void handleSignOut()}>
                  Sign out
                </Button>
              </div>
            ) : session.participant ? (
              <span className="shell-identity__role">Student</span>
            ) : (
              <Link to={loginPath()} className="shell-identity__link">
                Sign in
              </Link>
            )}
          </div>
        </div>
      </header>

      {courseId && !isAdmin && !anonymous && (
        <CourseContextBar
          name={metadata?.name}
          title={metadata?.title}
          term={metadata?.term}
          loading={metadataState.status === 'loading'}
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
