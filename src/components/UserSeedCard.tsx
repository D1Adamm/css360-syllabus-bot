import type { SeedExample } from '../types';
import { getSeedOriginLabel } from '../utils/seedDataUtils';

interface UserSeedCardProps {
  seed: SeedExample;
  onDelete: (id: string) => void | Promise<void>;
}

function formatComponentPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function UserSeedCard({ seed, onDelete }: UserSeedCardProps) {
  const canDelete = seed.origin === 'user';
  const originClassName =
    seed.origin === 'ai_generated'
      ? 'user-seed-card__label user-seed-card__label--origin user-seed-card__label--origin-ai'
      : 'user-seed-card__label user-seed-card__label--origin';
  const components = seed.validation?.components;
  const unsupportedClaims = seed.validation?.unsupportedClaims ?? [];

  async function handleDelete() {
    if (!canDelete) {
      return;
    }

    const confirmed = window.confirm(
      `Delete this example?\n\n"${seed.instruction}"\n\nThis cannot be undone.`,
    );

    if (confirmed) {
      await onDelete(seed.id);
    }
  }

  return (
    <article className="user-seed-card">
      <div className="user-seed-card__header">
        <h3 className="user-seed-card__question">{seed.instruction}</h3>
        <div className="user-seed-card__labels">
          <span className="user-seed-card__label user-seed-card__label--category">
            {seed.category}
          </span>
          <span
            className={`user-seed-card__label user-seed-card__label--difficulty user-seed-card__label--difficulty-${seed.difficulty.toLowerCase()}`}
          >
            {seed.difficulty}
          </span>
          <span className="user-seed-card__label user-seed-card__label--answer-type">
            {seed.directlyAnswered ? 'Directly answered' : 'Requires clarification'}
          </span>
          <span className={originClassName}>{getSeedOriginLabel(seed.origin)}</span>
          {seed.questionType && (
            <span className="user-seed-card__label user-seed-card__label--question-type">
              {seed.questionType}
            </span>
          )}
          {typeof seed.validation?.score === 'number' && (
            <span className="user-seed-card__label user-seed-card__label--validation">
              Validation {Math.round(seed.validation.score * 100)}%
            </span>
          )}
          {seed.status && seed.origin === 'ai_generated' && (
            <span className="user-seed-card__label user-seed-card__label--status">
              {seed.status}
            </span>
          )}
        </div>
      </div>

      <dl className="user-seed-card__meta">
        <div className="user-seed-card__meta-row">
          <dt>Source section</dt>
          <dd>{seed.sourceSection}</dd>
        </div>
        <div className="user-seed-card__meta-row">
          <dt>Origin</dt>
          <dd>{getSeedOriginLabel(seed.origin)}</dd>
        </div>
        {seed.questionType && (
          <div className="user-seed-card__meta-row">
            <dt>Question type</dt>
            <dd>{seed.questionType}</dd>
          </div>
        )}
        {seed.status && (
          <div className="user-seed-card__meta-row">
            <dt>Status</dt>
            <dd>{seed.status}</dd>
          </div>
        )}
        {seed.validation && (
          <div className="user-seed-card__meta-row">
            <dt>Validation</dt>
            <dd>
              {Math.round(seed.validation.score * 100)}% — {seed.validation.reason}
            </dd>
          </div>
        )}
        {components && (
          <div className="user-seed-card__meta-row">
            <dt>Component scores</dt>
            <dd>
              grounded {formatComponentPercent(components.grounded)}, correct{' '}
              {formatComponentPercent(components.correct)}, clear{' '}
              {formatComponentPercent(components.clear)}, useful{' '}
              {formatComponentPercent(components.useful)}, natural wording{' '}
              {formatComponentPercent(components.naturalStudentWording)}, category{' '}
              {formatComponentPercent(components.categoryCorrect)}, not trivial{' '}
              {formatComponentPercent(components.notTrivialOrTemporary)}
            </dd>
          </div>
        )}
        {unsupportedClaims.length > 0 && (
          <div className="user-seed-card__meta-row">
            <dt>Unsupported claims</dt>
            <dd>{unsupportedClaims.join('; ')}</dd>
          </div>
        )}
      </dl>

      {canDelete && (
        <button type="button" className="user-seed-card__delete" onClick={handleDelete}>
          Delete example
        </button>
      )}
    </article>
  );
}
