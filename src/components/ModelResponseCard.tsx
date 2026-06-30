import type { ComparisonResponse } from '../types';

interface ModelResponseCardProps {
  modelName: string;
  accessDescription: string;
  response: ComparisonResponse;
  isLoading?: boolean;
  error?: string | null;
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
}: ModelResponseCardProps) {
  return (
    <article
      className={`model-response-card${isLoading ? ' model-response-card--loading' : ''}${error ? ' model-response-card--error' : ''}`}
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

      {!isLoading && !error && (
        <p className="model-response-card__text">{response.text}</p>
      )}

      <footer className="model-response-card__footer">
        {!isLoading && !error && (
          <span className={groundingClassName(response.grounding)}>
            Grounding: {response.grounding}
          </span>
        )}
        {!isLoading && !error && response.simulated && (
          <span className="model-response-card__simulated">Simulated response</span>
        )}
        {!isLoading && !error && !response.simulated && (
          <span className="model-response-card__live">Live response</span>
        )}
      </footer>
    </article>
  );
}
