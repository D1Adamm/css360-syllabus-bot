import type { ComparisonResponse } from '../types';

interface ModelResponseCardProps {
  modelName: string;
  accessDescription: string;
  response: ComparisonResponse;
}

function groundingClassName(grounding: ComparisonResponse['grounding']): string {
  return `model-response-card__grounding model-response-card__grounding--${grounding.toLowerCase()}`;
}

export function ModelResponseCard({
  modelName,
  accessDescription,
  response,
}: ModelResponseCardProps) {
  return (
    <article className="model-response-card">
      <header className="model-response-card__header">
        <h3 className="model-response-card__title">{modelName}</h3>
        <p className="model-response-card__access">{accessDescription}</p>
      </header>

      <p className="model-response-card__text">{response.text}</p>

      <footer className="model-response-card__footer">
        <span className={groundingClassName(response.grounding)}>
          Grounding: {response.grounding}
        </span>
        {response.simulated && (
          <span className="model-response-card__simulated">Simulated response</span>
        )}
      </footer>
    </article>
  );
}
