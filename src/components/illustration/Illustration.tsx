import {
  getIllustrationSource,
  ILLUSTRATION_ALT,
  type IllustrationName,
} from '../../assets/illustrations';

export type IllustrationSize = 'sm' | 'md' | 'lg';

export interface IllustrationProps {
  name: IllustrationName;
  size?: IllustrationSize;
  /**
   * Illustrations are decorative by default and hidden from assistive
   * technology. Pass `decorative={false}` only when the image carries meaning
   * available nowhere else on the page — which it generally should not, since
   * the placeholder fallback cannot convey it.
   */
  decorative?: boolean;
  alt?: string;
  className?: string;
}

/**
 * Renders the illustration registered for `name`, or a quiet geometric
 * placeholder when no asset has been provided yet.
 *
 * Both branches occupy exactly the same box, so pages can be designed and
 * reviewed before any artwork exists and will not shift when it arrives.
 */
export function Illustration({
  name,
  size = 'md',
  decorative = true,
  alt,
  className,
}: IllustrationProps) {
  const source = getIllustrationSource(name);
  const classes = [`ui-illustration`, `ui-illustration--${size}`, className]
    .filter(Boolean)
    .join(' ');

  if (source) {
    return (
      <img
        src={source}
        className={classes}
        alt={decorative ? '' : (alt ?? ILLUSTRATION_ALT[name])}
        aria-hidden={decorative ? true : undefined}
        loading="lazy"
        draggable={false}
      />
    );
  }

  return (
    <div
      className={`${classes} ui-illustration--placeholder`}
      data-illustration={name}
      role={decorative ? undefined : 'img'}
      aria-label={decorative ? undefined : (alt ?? ILLUSTRATION_ALT[name])}
      aria-hidden={decorative ? true : undefined}
    >
      <svg viewBox="0 0 120 90" preserveAspectRatio="xMidYMid meet" focusable="false">
        <rect
          x="26"
          y="14"
          width="52"
          height="66"
          rx="4"
          className="ui-illustration__sheet"
        />
        <rect x="38" y="30" width="28" height="3" rx="1.5" className="ui-illustration__rule" />
        <rect x="38" y="40" width="22" height="3" rx="1.5" className="ui-illustration__rule" />
        <rect x="38" y="50" width="26" height="3" rx="1.5" className="ui-illustration__rule" />
        <circle cx="86" cy="56" r="16" className="ui-illustration__accent" />
      </svg>
    </div>
  );
}
