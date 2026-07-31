import type { ComparisonResponse } from '../types';

interface ModelResponseCardProps {
  modelName: string;
  accessDescription: string;
  response: ComparisonResponse;
  isLoading?: boolean;
  error?: string | null;
  sources?: string[];
  unavailableMessage?: string | null;
}

function groundingClassName(grounding: ComparisonResponse['grounding']): string {
  return `model-response-card__grounding model-response-card__grounding--${grounding.toLowerCase()}`;
}

export function ModelResponseCard({
  modelName,
  accessDescription,
  response,
  isLoading = false,
  error = null,
  sources = [],
  unavailableMessage = null,
}: ModelResponseCardProps) {
  const showUnavailable = !isLoading && !error && Boolean(unavailableMessage);

  return (
    <article
      className={`model-response-card${isLoading ? ' model-response-card--loading' : ''}${error ? ' model-response-card--error' : ''}${showUnavailable ? ' model-response-card--unavailable' : ''}`}
      aria-busy={isLoading}
    >
      <header className="model-response-card__header">
        <h3 className="model-response-card__title">{modelName}</h3>
        <p className="model-response-card__access">{accessDescription}</p>
      </header>

      {isLoading && (
        <p className="model-response-card__status" role="status">
          Generating response...
        </p>
      )}

      {!isLoading && error && (
        <p className="model-response-card__error" role="alert">
          {error}
        </p>
      )}

      {showUnavailable && (
        <p className="model-response-card__unavailable" role="status">
          {unavailableMessage}
        </p>
      )}

      {!isLoading && !error && !showUnavailable && (
        <p className="model-response-card__text">{response.text}</p>
      )}

      {!isLoading && !error && !showUnavailable && sources.length > 0 && (
        <div className="model-response-card__sources">
          <p className="model-response-card__sources-label">Syllabus sources</p>
          <ul className="model-response-card__sources-list">
            {sources.map((source, index) => (
              <li key={`${index}-${source}`}>{source}</li>
            ))}
          </ul>
        </div>
      )}

      <footer className="model-response-card__footer">
        {!isLoading && !error && !showUnavailable && (
          <span className={groundingClassName(response.grounding)}>
            Grounding: {response.grounding}
          </span>
        )}
        {!isLoading && !error && !showUnavailable && response.simulated && (
          <span className="model-response-card__simulated">Simulated response</span>
        )}
        {!isLoading && !error && !showUnavailable && !response.simulated && (
          <span className="model-response-card__live">Live response</span>
        )}
      </footer>
    </article>
  );
}
