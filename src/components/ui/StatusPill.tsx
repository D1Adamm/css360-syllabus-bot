export type StatusTone =
  | 'neutral'
  | 'info'
  | 'success'
  | 'warning'
  | 'danger'
  | 'accent'
  | 'progress';

export interface StatusPillProps {
  /**
   * `accent` is the gold state, reserved for "ready" outcomes.
   * `progress` is the in-flight state and animates gently (and not at all under
   * `prefers-reduced-motion`).
   */
  tone?: StatusTone;
  children: React.ReactNode;
  /** Hides the leading dot for pills used purely as quiet labels. */
  dot?: boolean;
  className?: string;
}

/**
 * A compact state label. Use sparingly — a screen covered in pills communicates
 * nothing. One per object, describing that object's single most important
 * state.
 */
export function StatusPill({
  tone = 'neutral',
  children,
  dot = true,
  className,
}: StatusPillProps) {
  const classes = ['ui-pill', `ui-pill--${tone}`, className].filter(Boolean).join(' ');

  return (
    <span className={classes}>
      {dot && <span className="ui-pill__dot" aria-hidden="true" />}
      {children}
    </span>
  );
}
