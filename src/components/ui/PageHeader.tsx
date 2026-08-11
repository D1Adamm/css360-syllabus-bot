export interface PageHeaderProps {
  title: string;
  /** A small line above the title: course identity, section, or context. */
  eyebrow?: React.ReactNode;
  /** One sentence. If it needs a paragraph, it belongs in the page body. */
  description?: string;
  /** Page-level actions, right-aligned on wide viewports. */
  actions?: React.ReactNode;
  className?: string;
}

/**
 * The top of a page.
 *
 * Title uses the display serif; everything else stays in the sans stack at a
 * modest size. Deliberately not a hero — this is application chrome, not
 * marketing.
 */
export function PageHeader({
  title,
  eyebrow,
  description,
  actions,
  className,
}: PageHeaderProps) {
  const classes = ['ui-page-header', className].filter(Boolean).join(' ');

  return (
    <header className={classes}>
      <div className="ui-page-header__text">
        {eyebrow && <p className="ui-page-header__eyebrow">{eyebrow}</p>}
        <h1 className="ui-page-header__title">{title}</h1>
        {description && <p className="ui-page-header__description">{description}</p>}
      </div>
      {actions && <div className="ui-page-header__actions">{actions}</div>}
    </header>
  );
}
