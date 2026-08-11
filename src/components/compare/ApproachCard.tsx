import { useId, useState } from 'react';
import type { ApproachState } from '../../hooks/useComparisonRun';
import { Icon } from '../ui/Icon';

export interface ApproachCardProps {
  label: string;
  description: string;
  state: ApproachState;
  /** Marker shown alongside the label so ratings can refer to a response. */
  marker?: string;
}

/**
 * One approach's answer.
 *
 * The response text is the largest, most prominent thing in the card. Where
 * the answer came from is real information but secondary, so it collapses into
 * a disclosure rather than competing with the answer for attention.
 */
export function ApproachCard({ label, description, state, marker }: ApproachCardProps) {
  const detailsId = useId();
  const [showSources, setShowSources] = useState(false);

  const sources = state.status === 'success' ? state.sources : [];

  return (
    <article
      className={`approach approach--${state.status}`}
      aria-busy={state.status === 'loading'}
    >
      <header className="approach__header">
        <h3 className="approach__label">
          {marker && <span className="approach__marker">{marker}</span>}
          {label}
        </h3>
        <p className="approach__description">{description}</p>
      </header>

      {state.status === 'idle' && (
        <p className="approach__placeholder">Ask a question to see this response.</p>
      )}

      {state.status === 'loading' && (
        <div className="approach__skeleton" role="status">
          <span className="ui-visually-hidden">Generating a response…</span>
          <span className="approach__skeleton-line" />
          <span className="approach__skeleton-line" />
          <span className="approach__skeleton-line approach__skeleton-line--short" />
        </div>
      )}

      {state.status === 'error' && (
        <p className="approach__unavailable">
          <Icon name="warning" size={15} />
          {state.message}
        </p>
      )}

      {state.status === 'success' && <p className="approach__text">{state.text}</p>}

      {state.status === 'success' && sources.length > 0 && (
        <div className="approach__sources">
          <button
            type="button"
            className="approach__disclosure"
            aria-expanded={showSources}
            aria-controls={detailsId}
            onClick={() => setShowSources((open) => !open)}
          >
            <Icon name="expand" size={14} />
            Where this came from
            <span className="approach__source-count">{sources.length}</span>
          </button>
          {showSources && (
            <ul id={detailsId} className="approach__source-list">
              {sources.map((source, index) => (
                <li key={`${index}-${source}`}>{source}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </article>
  );
}
