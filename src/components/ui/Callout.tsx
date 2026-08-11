import { Icon, type IconName } from './Icon';

export type CalloutTone = 'info' | 'success' | 'warning' | 'danger';

const TONE_ICON: Record<CalloutTone, IconName> = {
  info: 'info',
  success: 'success',
  warning: 'warning',
  danger: 'warning',
};

export interface CalloutProps {
  tone?: CalloutTone;
  title?: string;
  /** Actions render beneath the message, e.g. a retry button. */
  actions?: React.ReactNode;
  children?: React.ReactNode;
  className?: string;
  /**
   * Live-region behaviour. `danger` announces assertively as an alert;
   * everything else announces politely as a status. Pass `false` for a purely
   * static note that should not be announced at all.
   */
  live?: boolean;
}

/**
 * One component for every inline message in the application.
 *
 * This replaces the seven bespoke notice classes in the legacy stylesheet
 * (`syllabus-notice`, `dataset-notice`, `comparison-notice`,
 * `evaluation-notice`, `results-notice`, `seed-builder-notice`,
 * `prototype-banner`), which were all the same idea styled seven times.
 */
export function Callout({
  tone = 'info',
  title,
  actions,
  children,
  className,
  live = true,
}: CalloutProps) {
  const classes = ['ui-callout', `ui-callout--${tone}`, className]
    .filter(Boolean)
    .join(' ');

  const liveProps = live
    ? tone === 'danger'
      ? ({ role: 'alert' } as const)
      : ({ role: 'status', 'aria-live': 'polite' } as const)
    : {};

  return (
    <div className={classes} {...liveProps}>
      <Icon name={TONE_ICON[tone]} size={16} className="ui-callout__icon" />
      <div className="ui-callout__body">
        {title && <p className="ui-callout__title">{title}</p>}
        {children && <div className="ui-callout__message">{children}</div>}
        {actions && <div className="ui-callout__actions">{actions}</div>}
      </div>
    </div>
  );
}
