import type { SeedExample } from '../types';
import { UserSeedCard } from './UserSeedCard';

interface UserSeedListProps {
  seeds: SeedExample[];
  onDelete: (id: string) => void | Promise<void>;
  onDeleteAll: () => void | Promise<void>;
}

function isManualSeed(seed: SeedExample): boolean {
  return seed.origin === 'user';
}

export function UserSeedList({ seeds, onDelete, onDeleteAll }: UserSeedListProps) {
  const manualSeeds = seeds.filter(isManualSeed);

  async function handleDeleteAll() {
    const confirmed = window.confirm(
      'Delete all your examples? This removes your manually created examples and cannot be undone. AI-generated seeds in Review/Dataset are left unchanged.',
    );

    if (confirmed) {
      await onDeleteAll();
    }
  }

  return (
    <section className="user-seed-list" aria-labelledby="user-seed-list-title">
      <div className="user-seed-list__header">
        <h2 id="user-seed-list-title" className="user-seed-list__title">
          Your seed examples ({manualSeeds.length})
        </h2>
        {manualSeeds.length > 0 && (
          <button
            type="button"
            className="user-seed-list__delete-all"
            onClick={handleDeleteAll}
          >
            Delete all my examples
          </button>
        )}
      </div>

      {manualSeeds.length === 0 ? (
        <p className="user-seed-list__empty">
          No manually created examples yet. Use the form to add one. AI-generated and reviewed
          seeds appear on Review Seeds and Dataset instead.
        </p>
      ) : (
        <ul className="user-seed-list__items">
          {manualSeeds.map((seed) => (
            <li key={seed.id}>
              <UserSeedCard seed={seed} onDelete={onDelete} />
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
