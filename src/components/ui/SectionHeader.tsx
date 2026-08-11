export interface SectionHeaderProps {
  title: string;
  description?: string;
  actions?: React.ReactNode;
  /** Heading level. Keep the document outline correct; do not pick by size. */
  level?: 2 | 3;
  /** Adds a hairline beneath the header instead of wrapping in a box. */
  divider?: boolean;
  id?: string;
  className?: string;
}

/** A heading within a page, with optional inline actions. */
export function SectionHeader({
  title,
  description,
  actions,
  level = 2,
  divider = false,
  id,
  className,
}: SectionHeaderProps) {
  const classes = [
    'ui-section-header',
    divider ? 'ui-section-header--divider' : '',
    className,
  ]
    .filter(Boolean)
    .join(' ');

  const Heading = level === 2 ? 'h2' : 'h3';

  return (
    <div className={classes}>
      <div className="ui-section-header__text">
        <Heading className="ui-section-header__title" id={id}>
          {title}
        </Heading>
        {description && <p className="ui-section-header__description">{description}</p>}
      </div>
      {actions && <div className="ui-section-header__actions">{actions}</div>}
    </div>
  );
}
