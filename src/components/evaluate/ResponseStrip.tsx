import { useState } from 'react';
import { APPROACHES } from '../compare/approaches';
import type { ComparisonRun } from '../../context/comparisonRun';
import { Icon } from '../ui/Icon';

export interface ResponseStripProps {
  run: ComparisonRun;
}

/**
 * The four responses being rated, kept on screen while rating.
 *
 * Every criterion is comparative ("which was most accurate"), so the answers
 * have to stay visible — asking someone to remember four paragraphs while
 * choosing between them is how you get noise instead of data. Each response
 * carries the same A–D marker used by the rating controls below.
 */
export function ResponseStrip({ run }: ResponseStripProps) {
  const [expanded, setExpanded] = useState(true);

  return (
    <section className="strip" aria-label="Responses you are rating">
      <div className="strip__header">
        <p className="strip__question">
          <span className="strip__question-label">You asked</span>
          {run.question}
        </p>
        <button
          type="button"
          className="strip__toggle"
          aria-expanded={expanded}
          onClick={() => setExpanded((open) => !open)}
        >
          <Icon name="expand" size={14} />
          {expanded ? 'Hide responses' : 'Show responses'}
        </button>
      </div>

      {expanded && (
        <div className="strip__grid">
          {APPROACHES.map((approach, index) => {
            const response = run.responses[approach.key];
            const marker = String.fromCharCode(65 + index);

            return (
              <article key={approach.key} className="strip__card">
                <h3 className="strip__card-label">
                  <span className="strip__marker">{marker}</span>
                  {approach.label}
                </h3>
                <p className="strip__card-description">{approach.description}</p>
                {response.error ? (
                  <p className="strip__card-unavailable">{response.error}</p>
                ) : (
                  <p className="strip__card-text">{response.text}</p>
                )}
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
