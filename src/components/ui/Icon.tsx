import { ICON_REGISTRY, type IconName } from './icons';

export type { IconName } from './icons';

export interface IconProps {
  name: IconName;
  /** Pixel size. Defaults to 16 so icons sit with body text without shouting. */
  size?: number;
  strokeWidth?: number;
  className?: string;
  /**
   * Accessible label. Omit it when the icon merely decorates adjacent text —
   * the icon is then hidden from assistive technology, which is the common and
   * correct case. Provide it only when the icon is the sole carrier of meaning.
   */
  label?: string;
}

/**
 * Renders an icon by product meaning rather than by vendor name.
 *
 * The icon library is referenced only from `icons.ts`, so swapping it or
 * re-choosing a glyph touches exactly one file.
 */
export function Icon({
  name,
  size = 16,
  strokeWidth = 1.75,
  className,
  label,
}: IconProps) {
  const Glyph = ICON_REGISTRY[name];

  return (
    <Glyph
      size={size}
      strokeWidth={strokeWidth}
      className={className ? `ui-icon ${className}` : 'ui-icon'}
      aria-hidden={label ? undefined : true}
      aria-label={label}
      role={label ? 'img' : undefined}
      focusable="false"
    />
  );
}
