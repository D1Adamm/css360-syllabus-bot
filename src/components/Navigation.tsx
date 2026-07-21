import { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { coursePagePath, type CoursePageSegment } from '../lib/courseRoutes';

interface NavItem {
  to: string;
  label: string;
  end?: boolean;
}

function buildNavItems(courseId: string | null): NavItem[] {
  const shared: NavItem[] = [
    { to: '/', label: 'Courses', end: true },
    { to: '/create-course', label: 'Create Course' },
    { to: '/architecture', label: 'Architecture' },
  ];

  if (!courseId) {
    return shared;
  }

  const courseLink = (segment: CoursePageSegment, label: string, end?: boolean): NavItem => ({
    to: coursePagePath(courseId, segment),
    label,
    end,
  });

  return [
    courseLink('home', 'Home', true),
    courseLink('syllabus', 'Syllabus'),
    courseLink('seeds', 'Build Seeds'),
    courseLink('review', 'Review Seeds'),
    courseLink('dataset', 'Dataset'),
    courseLink('compare', 'Compare'),
    courseLink('evaluate', 'Evaluate'),
    courseLink('results', 'Results'),
    ...shared,
  ];
}

function getNavLinkClass(isActive: boolean): string {
  return isActive ? 'nav-link nav-link--active' : 'nav-link';
}

interface NavigationProps {
  courseId: string | null;
}

export function Navigation({ courseId }: NavigationProps) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const navItems = buildNavItems(courseId);

  function closeMobile() {
    setMobileOpen(false);
  }

  function toggleMobile() {
    setMobileOpen((open) => !open);
  }

  return (
    <>
      <nav className="nav-desktop" aria-label="Main navigation">
        <ul className="nav-list">
          {navItems.map(({ to, label, end }) => (
            <li key={`${label}-${to}`}>
              <NavLink
                to={to}
                end={end}
                className={({ isActive }) => getNavLinkClass(isActive)}
              >
                {label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      <button
        type="button"
        className="nav-mobile-toggle"
        aria-expanded={mobileOpen}
        aria-controls="mobile-nav"
        aria-label="Toggle navigation menu"
        onClick={toggleMobile}
      >
        Menu
      </button>

      <nav
        id="mobile-nav"
        className={`nav-mobile${mobileOpen ? ' nav-mobile--open' : ''}`}
        aria-label="Mobile navigation"
      >
        <ul className="nav-list nav-list--mobile">
          {navItems.map(({ to, label, end }) => (
            <li key={`${label}-${to}`}>
              <NavLink
                to={to}
                end={end}
                className={({ isActive }) => getNavLinkClass(isActive)}
                onClick={closeMobile}
              >
                {label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </>
  );
}
