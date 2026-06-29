import { useId, useState } from 'react';
import type { SeedExample } from '../types';

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

function getOriginLabel(origin: SeedExample['origin']): string {
  return origin === 'user' ? 'User created' : 'Prototype generated';
}

export function SeedCard({ seed, onDelete }: SeedCardProps) {
  const [expanded, setExpanded] = useState(false);
  const detailsId = useId();
  const preview = truncateText(seed.response, PREVIEW_LENGTH);
  const isUserCreated = seed.origin === 'user';

  function handleDelete() {
    if (!onDelete) {
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
          <span
            className={`seed-card__origin${
              isUserCreated ? ' seed-card__origin--user' : ''
            }`}
          >
            {getOriginLabel(seed.origin)}
          </span>
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
            <span className="seed-card__meta-label">Origin:</span> {getOriginLabel(seed.origin)}
          </p>
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

        {isUserCreated && onDelete && (
          <button type="button" className="seed-card__delete" onClick={handleDelete}>
            Delete example
          </button>
        )}
      </div>
    </article>
  );
}
