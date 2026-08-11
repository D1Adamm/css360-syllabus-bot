import { Link } from 'react-router-dom';

export interface BrandMarkProps {
  to: string;
}

/**
 * Product name plus institutional context.
 *
 * The institutional line names the school this research sits in without
 * borrowing university branding or implying this is an official University of
 * Washington service. The footer states that outright.
 */
export function BrandMark({ to }: BrandMarkProps) {
  return (
    <Link to={to} className="shell-brand">
      <span className="shell-brand__mark" aria-hidden="true" />
      <span className="shell-brand__text">
        <span className="shell-brand__name">Syllabus Model Lab</span>
        <span className="shell-brand__org">UW Bothell · School of STEM</span>
      </span>
    </Link>
  );
}
