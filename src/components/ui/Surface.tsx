import { createElement } from 'react';

export type SurfaceTone = 'plain' | 'raised' | 'sunken' | 'accent';
export type SurfacePadding = 'none' | 'sm' | 'md' | 'lg';

export interface SurfaceProps {
  /**
   * `plain` is transparent and unbordered — the default on purpose. Most
   * sections need grouping and whitespace, not another rectangle. Reach for
   * `raised` only when a thing genuinely needs to read as a discrete object.
   */
  tone?: SurfaceTone;
  padding?: SurfacePadding;
  /** Adds a hairline border. Implied by `raised` and `sunken`. */
  bordered?: boolean;
  as?: 'div' | 'section' | 'article' | 'aside' | 'li' | 'form';
  className?: string;
  children?: React.ReactNode;
  id?: string;
  'aria-label'?: string;
  'aria-labelledby'?: string;
  'aria-live'?: 'polite' | 'assertive' | 'off';
}

/**
 * A neutral container. Deliberately boring: it owns background, padding and an
 * optional hairline, and nothing else. Composition and spacing come from the
 * `ui-stack` / `ui-row` utilities.
 */
export function Surface({
  tone = 'plain',
  padding = 'none',
  bordered = false,
  as = 'div',
  className,
  children,
  ...rest
}: SurfaceProps) {
  const classes = [
    'ui-surface',
    `ui-surface--${tone}`,
    padding !== 'none' ? `ui-surface--pad-${padding}` : '',
    bordered ? 'ui-surface--bordered' : '',
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return createElement(as, { className: classes, ...rest }, children);
}
