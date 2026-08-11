import { useEffect, useId, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { Icon } from '../components/ui/Icon';
import type { NavItem } from './navigation';

export interface PrimaryNavProps {
  items: NavItem[];
  label?: string;
  /**
   * When false the list is always visible and no Menu toggle renders. The
   * admin sidebar uses this: it already stacks, so collapsing it would hide
   * navigation behind an extra tap for no benefit.
   */
  collapsible?: boolean;
}

/**
 * One rendered list, restyled at narrow widths.
 *
 * At desktop widths it is a horizontal bar; below 48rem it collapses behind a
 * Menu button and becomes a panel. The markup is identical in both cases, so
 * there is no second copy to keep in sync and no hidden duplicate for screen
 * readers to announce twice.
 */
export function PrimaryNav({
  items,
  label = 'Main navigation',
  collapsible = true,
}: PrimaryNavProps) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const { pathname } = useLocation();

  // Navigating away should always close the menu.
  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  if (items.length === 0) {
    return null;
  }

  return (
    <div className={collapsible ? 'shell-nav' : 'shell-nav shell-nav--static'}>
      {collapsible && (
        <button
          type="button"
          className="shell-nav__toggle"
          aria-expanded={open}
          aria-controls={panelId}
          onClick={() => setOpen((current) => !current)}
        >
          <Icon name={open ? 'error' : 'menu'} size={16} />
          <span>Menu</span>
        </button>
      )}

      <nav
        id={panelId}
        className={`shell-nav__panel${open ? ' shell-nav__panel--open' : ''}`}
        aria-label={label}
      >
        <ul className="shell-nav__list">
          {items.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  isActive ? 'shell-nav__link shell-nav__link--active' : 'shell-nav__link'
                }
              >
                <Icon name={item.icon} size={16} className="shell-nav__icon" />
                <span>{item.label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  );
}
