import { NavLink } from 'react-router-dom';
import { Icon } from '../components/ui/Icon';
import { formatCourseCode } from '../lib/courseLabels';
import type { NavItem } from './navigation';

export interface CourseContextBarProps {
  /** Course code, e.g. "CSS 360". Absent until the record has been read. */
  name?: string;
  title?: string;
  term?: string;
  /** True while the course record is still being read. */
  loading?: boolean;
  items: NavItem[];
}

/**
 * A compact strip identifying the active course, with its secondary links.
 *
 * Kept to one line so the shell stays shallow — course identity belongs here,
 * not in a tall banner repeated on every page.
 */
export function CourseContextBar({
  name,
  title,
  term,
  loading = false,
  items,
}: CourseContextBarProps) {
  const code = formatCourseCode(name) || name;

  return (
    <div className="shell-course">
      <div className="ui-container shell-course__inner">
        {/*
         * While the record is loading we show a placeholder of the same size
         * rather than the word "Course". Rendering a stand-in name meant the
         * bar briefly asserted something false about the course, and the
         * correction one tick later read as a glitch. A neutral shape says
         * "not known yet", which is the truth.
         */}
        <div className="shell-course__identity">
          {loading ? (
            <span
              className="shell-course__name-skeleton"
              aria-hidden="true"
              data-testid="course-identity-skeleton"
            />
          ) : (
            <>
              <span className="shell-course__name">{code}</span>
              {title && <span className="shell-course__title">{title}</span>}
              {term && <span className="shell-course__term">{term}</span>}
            </>
          )}
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
