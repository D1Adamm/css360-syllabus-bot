import type { IllustrationName } from '../../assets/illustrations';
import { Illustration } from '../illustration/Illustration';

export interface EmptyStateProps {
  title: string;
  description?: string;
  /** Optional illustration slot. Falls back to a placeholder when unset. */
  illustration?: IllustrationName;
  /** Primary (and at most one secondary) action. */
  action?: React.ReactNode;
  /** `compact` suits an empty list inside a page; `full` suits a whole page. */
  size?: 'compact' | 'full';
  className?: string;
}

/**
 * The state a screen shows before it has anything to show.
 *
 * Treated as a first-class design surface rather than an afterthought: it is
 * usually the first thing a new student or professor sees, and it is where the
 * product explains itself.
 */
export function EmptyState({
  title,
  description,
  illustration,
  action,
  size = 'compact',
  className,
}: EmptyStateProps) {
  const classes = ['ui-empty', `ui-empty--${size}`, className].filter(Boolean).join(' ');

  return (
    <div className={classes}>
      {illustration && (
        <Illustration name={illustration} size={size === 'full' ? 'lg' : 'md'} />
      )}
      <div className="ui-empty__text">
        <p className="ui-empty__title">{title}</p>
        {description && <p className="ui-empty__description">{description}</p>}
      </div>
      {action && <div className="ui-empty__action">{action}</div>}
    </div>
  );
}
