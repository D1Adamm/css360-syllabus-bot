import { useState } from 'react';
import { Button } from './Button';
import { Icon } from './Icon';

export interface ErrorStateProps {
  title: string;
  message: string;
  /** Wire this up whenever the request can simply be tried again. */
  onRetry?: () => void;
  retryLabel?: string;
  /**
   * Raw technical text. Admin only — it renders behind a disclosure so it
   * never competes with the message. Never pass this on a student or
   * professor surface.
   */
  technical?: string;
  className?: string;
}

/**
 * A failed request, shown so it cannot be mistaken for "there is no data".
 *
 * This distinction matters more than it looks: a page that renders an error
 * banner *and* a set of zero counts is telling the reader two contradictory
 * things, and the zeros are the more believable of the two. Wherever this
 * component appears, the data it replaces must not render at all.
 */
export function ErrorState({
  title,
  message,
  onRetry,
  retryLabel = 'Try again',
  technical,
  className,
}: ErrorStateProps) {
  const [showTechnical, setShowTechnical] = useState(false);
  const classes = ['error-state', className].filter(Boolean).join(' ');

  return (
    <div className={classes} role="alert">
      <Icon name="warning" size={20} className="error-state__icon" />
      <div className="error-state__body">
        <p className="error-state__title">{title}</p>
        <p className="error-state__message">{message}</p>

        <div className="error-state__actions">
          {onRetry && (
            <Button variant="secondary" size="sm" onClick={onRetry} iconLeft="status">
              {retryLabel}
            </Button>
          )}
          {technical && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowTechnical((open) => !open)}
              aria-expanded={showTechnical}
            >
              {showTechnical ? 'Hide detail' : 'Show detail'}
            </Button>
          )}
        </div>

        {technical && showTechnical && (
          <pre className="error-state__technical">{technical}</pre>
        )}
      </div>
    </div>
  );
}
