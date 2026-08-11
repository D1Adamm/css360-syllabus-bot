import { Link, type LinkProps } from 'react-router-dom';
import { Icon, type IconName } from './Icon';

/**
 * Action hierarchy. Exactly one `primary` per view — purple is the signal that
 * something is *the* thing to do, and it stops meaning that if every button
 * uses it.
 *
 *   primary   filled purple      the single main action
 *   secondary outlined           common alternatives
 *   tertiary  text + hairline    low-emphasis, sits in rows of actions
 *   ghost     text only          toolbar / inline affordances
 *   danger    outlined red       destructive; confirmation still required
 */
export type ButtonVariant =
  | 'primary'
  | 'secondary'
  | 'tertiary'
  | 'ghost'
  | 'danger';

export type ButtonSize = 'sm' | 'md';

interface ButtonAppearance {
  variant?: ButtonVariant;
  size?: ButtonSize;
  iconLeft?: IconName;
  iconRight?: IconName;
  fullWidth?: boolean;
}

function appearanceClass({
  variant = 'secondary',
  size = 'md',
  fullWidth = false,
}: ButtonAppearance): string {
  return [
    'ui-button',
    `ui-button--${variant}`,
    `ui-button--${size}`,
    fullWidth ? 'ui-button--full' : '',
  ]
    .filter(Boolean)
    .join(' ');
}

export interface ButtonProps
  extends ButtonAppearance,
    React.ButtonHTMLAttributes<HTMLButtonElement> {
  /** Shows a busy state and blocks activation without changing layout width. */
  loading?: boolean;
  loadingLabel?: string;
  /** React 19 passes `ref` as an ordinary prop; no `forwardRef` needed. */
  ref?: React.Ref<HTMLButtonElement>;
}

export function Button({
  variant,
  size,
  iconLeft,
  iconRight,
  fullWidth,
  loading = false,
  loadingLabel,
  disabled,
  children,
  className,
  type = 'button',
  ref,
  ...rest
}: ButtonProps) {
  const classes = [appearanceClass({ variant, size, fullWidth }), className]
    .filter(Boolean)
    .join(' ');

  return (
    <button
      {...rest}
      ref={ref}
      type={type}
      className={classes}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
    >
      {iconLeft && !loading && <Icon name={iconLeft} size={size === 'sm' ? 14 : 16} />}
      {loading && <span className="ui-button__spinner" aria-hidden="true" />}
      <span className="ui-button__label">
        {loading && loadingLabel ? loadingLabel : children}
      </span>
      {iconRight && !loading && <Icon name={iconRight} size={size === 'sm' ? 14 : 16} />}
    </button>
  );
}

export interface LinkButtonProps extends ButtonAppearance, LinkProps {}

/** A router link that presents as a button. Same hierarchy rules apply. */
export function LinkButton({
  variant,
  size,
  iconLeft,
  iconRight,
  fullWidth,
  children,
  className,
  ...rest
}: LinkButtonProps) {
  const classes = [appearanceClass({ variant, size, fullWidth }), className]
    .filter(Boolean)
    .join(' ');

  return (
    <Link {...rest} className={classes}>
      {iconLeft && <Icon name={iconLeft} size={size === 'sm' ? 14 : 16} />}
      <span className="ui-button__label">{children}</span>
      {iconRight && <Icon name={iconRight} size={size === 'sm' ? 14 : 16} />}
    </Link>
  );
}
