import type { ModelKey } from '../types';
import { getModelLabel, MODEL_KEYS } from '../utils/evaluationUtils';

interface ModelBarChartProps {
  title: string;
  counts: Record<ModelKey, number>;
  total: number;
  id: string;
}

export function ModelBarChart({ title, counts, total, id }: ModelBarChartProps) {
  return (
    <section className="results-chart" aria-labelledby={id}>
      <h3 id={id} className="results-chart__title">
        {title}
      </h3>
      <ul className="results-chart__list" role="list">
        {MODEL_KEYS.map((key) => {
          const count = counts[key];
          const percentage = total > 0 ? Math.round((count / total) * 100) : 0;
          const barId = `${id}-bar-${key}`;
          return (
            <li key={key} className="results-chart__item">
              <div className="results-chart__label-row">
                <span className="results-chart__label">{getModelLabel(key)}</span>
                <span className="results-chart__value" aria-hidden="true">
                  {count} ({percentage}%)
                </span>
              </div>
              <div
                className="results-chart__track"
                role="img"
                aria-label={`${getModelLabel(key)}: ${count} of ${total}, ${percentage} percent`}
              >
                <div
                  id={barId}
                  className="results-chart__bar"
                  style={{ width: `${percentage}%` }}
                />
              </div>
              <span className="ui-visually-hidden">
                {getModelLabel(key)}: {count} votes, {percentage} percent
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
