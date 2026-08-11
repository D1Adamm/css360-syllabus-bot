import { Link } from 'react-router-dom';
import { Icon } from '../ui/Icon';
import { formatCourseCode } from '../../lib/courseLabels';

export interface CourseRowProps {
  to: string;
  name: string;
  title?: string;
  /** A short factual line: term, instructor, ids for admin. */
  meta?: React.ReactNode;
  /** Status pill or similar, right-aligned before the chevron. */
  status?: React.ReactNode;
  /** Extra technical detail rendered under the meta line (admin only). */
  detail?: React.ReactNode;
}

/**
 * One openable course in a list.
 *
 * The whole row is the link rather than just the name. A list of course names
 * with a clickable word in it reads as text; a row that highlights and shows a
 * chevron reads as something you can open. Using a single `<Link>` for the
 * whole row keeps that behaviour without inventing click handlers on divs, so
 * keyboard focus, middle-click, and open-in-new-tab all work as expected.
 */
export function CourseRow({ to, name, title, meta, status, detail }: CourseRowProps) {
  const code = formatCourseCode(name);
  // Without this the link announces every scrap of text in the row as one
  // run-on name. "CSS 350, Management Principles" is what a listener needs.
  const label = title ? `${code}, ${title}` : code;

  return (
    <li className="course-row">
      <Link to={to} className="course-row__link" aria-label={label}>
        <span className="course-row__main">
          <span className="course-row__name">{code}</span>
          {title && <span className="course-row__title">{title}</span>}
          {meta && <span className="course-row__meta">{meta}</span>}
          {detail && <span className="course-row__detail">{detail}</span>}
        </span>

        {status && <span className="course-row__status">{status}</span>}

        <Icon name="next" size={18} className="course-row__chevron" />
      </Link>
    </li>
  );
}
