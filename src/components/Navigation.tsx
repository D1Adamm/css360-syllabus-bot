import { useState } from 'react';
import { NavLink } from 'react-router-dom';

const navItems = [
  { to: '/', label: 'Home', end: true },
  { to: '/syllabus', label: 'Syllabus' },
  { to: '/seed-builder', label: 'Build Seeds' },
  { to: '/dataset', label: 'Dataset' },
  { to: '/compare', label: 'Compare' },
  { to: '/evaluate', label: 'Evaluate' },
  { to: '/results', label: 'Results' },
  { to: '/architecture', label: 'Architecture' },
] as const;

function getNavLinkClass(isActive: boolean): string {
  return isActive ? 'nav-link nav-link--active' : 'nav-link';
}

export function Navigation() {
  const [mobileOpen, setMobileOpen] = useState(false);

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
          {navItems.map(({ to, label, ...rest }) => (
            <li key={to}>
              <NavLink
                to={to}
                className={({ isActive }) => getNavLinkClass(isActive)}
                {...rest}
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
          {navItems.map(({ to, label, ...rest }) => (
            <li key={to}>
              <NavLink
                to={to}
                className={({ isActive }) => getNavLinkClass(isActive)}
                onClick={closeMobile}
                {...rest}
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
