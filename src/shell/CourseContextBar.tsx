import { NavLink } from 'react-router-dom';
import { Icon } from '../components/ui/Icon';
import { formatCourseCode } from '../lib/courseLabels';
import type { NavItem } from './navigation';

export interface CourseContextBarProps {
  /** Course code, e.g. "CSS 360". Falls back to the id while metadata loads. */
  name: string;
  title?: string;
  term?: string;
  items: NavItem[];
}

/**
 * A compact strip identifying the active course, with its secondary links.
 *
 * Kept to one line so the shell stays shallow — course identity belongs here,
 * not in a tall banner repeated on every page.
 */
export function CourseContextBar({ name, title, term, items }: CourseContextBarProps) {
  return (
    <div className="shell-course">
      <div className="ui-container shell-course__inner">
        <div className="shell-course__identity">
          <span className="shell-course__name">{formatCourseCode(name) || name}</span>
          {title && <span className="shell-course__title">{title}</span>}
          {term && <span className="shell-course__term">{term}</span>}
        </div>

        {items.length > 0 && (
          <nav className="shell-course__nav" aria-label="Course sections">
            <ul className="shell-course__list">
              {items.map((item) => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    end={item.end}
                    className={({ isActive }) =>
                      isActive
                        ? 'shell-course__link shell-course__link--active'
                        : 'shell-course__link'
                    }
                  >
                    <Icon name={item.icon} size={14} />
                    <span>{item.label}</span>
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
        )}
      </div>
    </div>
  );
}
