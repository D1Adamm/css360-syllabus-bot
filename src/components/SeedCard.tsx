import { useId, useState } from 'react';
import type { SeedExample } from '../types';
import { getSeedOriginLabel } from '../utils/seedDataUtils';

interface SeedCardProps {
  seed: SeedExample;
  onDelete?: (id: string) => void;
}

const PREVIEW_LENGTH = 120;

function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) {
    return text;
  }

  return `${text.slice(0, maxLength).trimEnd()}…`;
}

function formatComponentPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function SeedCard({ seed, onDelete }: SeedCardProps) {
  const [expanded, setExpanded] = useState(false);
  const detailsId = useId();
  const preview = truncateText(seed.response, PREVIEW_LENGTH);
  const canDelete = seed.origin === 'user';
  const originClassName =
    seed.origin === 'user'
      ? 'seed-card__origin seed-card__origin--user'
      : seed.origin === 'ai_generated'
        ? 'seed-card__origin seed-card__origin--ai'
        : 'seed-card__origin';
  const components = seed.validation?.components;
  const unsupportedClaims = seed.validation?.unsupportedClaims ?? [];

  function handleDelete() {
    if (!onDelete || !canDelete) {
      return;
    }

    const confirmed = window.confirm(
      `Delete this example?\n\n"${seed.instruction}"\n\nThis cannot be undone.`,
    );

    if (confirmed) {
      onDelete(seed.id);
    }
  }

  return (
    <article className="seed-card">
      <div className="seed-card__header">
        <div className="seed-card__labels">
          <span className="seed-card__category">{seed.category}</span>
          <span
            className={`seed-card__difficulty seed-card__difficulty--${seed.difficulty.toLowerCase()}`}
          >
            {seed.difficulty}
          </span>
          <span
            className={`seed-card__answer-type${
              seed.directlyAnswered ? '' : ' seed-card__answer-type--clarification'
            }`}
          >
            {seed.directlyAnswered ? 'Directly answered' : 'Requires clarification'}
          </span>
          <span className={originClassName}>{getSeedOriginLabel(seed.origin)}</span>
          {seed.questionType && (
            <span className="seed-card__question-type">{seed.questionType}</span>
          )}
          {typeof seed.validation?.score === 'number' && (
            <span className="seed-card__validation">
              Validation {Math.round(seed.validation.score * 100)}%
            </span>
          )}
          {seed.status && seed.origin === 'ai_generated' && (
            <span className="seed-card__status">{seed.status}</span>
          )}
        </div>
        <h2 className="seed-card__instruction">{seed.instruction}</h2>
      </div>

      <p className="seed-card__response">
        {expanded ? seed.response : preview}
      </p>

      {expanded && (
        <div id={detailsId} className="seed-card__metadata">
          <p className="seed-card__meta-item">
            <span className="seed-card__meta-label">Source section:</span>{' '}
            {seed.sourceSection}
          </p>
          <p className="seed-card__meta-item">
            <span className="seed-card__meta-label">ID:</span> {seed.id}
          </p>
          <p className="seed-card__meta-item">
            <span className="seed-card__meta-label">Origin:</span>{' '}
            {getSeedOriginLabel(seed.origin)}
          </p>
          {seed.questionType && (
            <p className="seed-card__meta-item">
              <span className="seed-card__meta-label">Question type:</span>{' '}
              {seed.questionType}
            </p>
          )}
          {seed.status && (
            <p className="seed-card__meta-item">
              <span className="seed-card__meta-label">Status:</span> {seed.status}
            </p>
          )}
          {seed.validation && (
            <p className="seed-card__meta-item">
              <span className="seed-card__meta-label">Validation:</span>{' '}
              {Math.round(seed.validation.score * 100)}% — {seed.validation.reason}
            </p>
          )}
          {components && (
            <p className="seed-card__meta-item">
              <span className="seed-card__meta-label">Component scores:</span>{' '}
              grounded {formatComponentPercent(components.grounded)}, correct{' '}
              {formatComponentPercent(components.correct)}, clear{' '}
              {formatComponentPercent(components.clear)}, useful{' '}
              {formatComponentPercent(components.useful)}, natural wording{' '}
              {formatComponentPercent(components.naturalStudentWording)}, category{' '}
              {formatComponentPercent(components.categoryCorrect)}, not trivial{' '}
              {formatComponentPercent(components.notTrivialOrTemporary)}
            </p>
          )}
          {unsupportedClaims.length > 0 && (
            <p className="seed-card__meta-item">
              <span className="seed-card__meta-label">Unsupported claims:</span>{' '}
              {unsupportedClaims.join('; ')}
            </p>
          )}
          {seed.notes && (
            <p className="seed-card__meta-item">
              <span className="seed-card__meta-label">Notes:</span> {seed.notes}
            </p>
          )}
        </div>
      )}

      <div className="seed-card__actions">
        <button
          type="button"
          className="seed-card__toggle"
          aria-expanded={expanded}
          aria-controls={detailsId}
          onClick={() => setExpanded((current) => !current)}
        >
          {expanded ? 'Collapse example' : 'Expand example'}
        </button>

        {canDelete && onDelete && (
          <button type="button" className="seed-card__delete" onClick={handleDelete}>
            Delete example
          </button>
        )}
      </div>
    </article>
  );
}
