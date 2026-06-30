import type { SeedExample } from '../types';

interface UserSeedCardProps {
  seed: SeedExample;
  onDelete: (id: string) => void | Promise<void>;
}

export function UserSeedCard({ seed, onDelete }: UserSeedCardProps) {
  async function handleDelete() {
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
          <span className="user-seed-card__label user-seed-card__label--origin">
            User created
          </span>
        </div>
      </div>

      <dl className="user-seed-card__meta">
        <div className="user-seed-card__meta-row">
          <dt>Source section</dt>
          <dd>{seed.sourceSection}</dd>
        </div>
        <div className="user-seed-card__meta-row">
          <dt>Origin</dt>
          <dd>{seed.origin}</dd>
        </div>
      </dl>

      <button type="button" className="user-seed-card__delete" onClick={handleDelete}>
        Delete example
      </button>
    </article>
  );
}
