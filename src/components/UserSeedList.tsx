import type { SeedExample } from '../types';
import { UserSeedCard } from './UserSeedCard';

interface UserSeedListProps {
  seeds: SeedExample[];
  onDelete: (id: string) => void;
  onDeleteAll: () => void;
}

export function UserSeedList({ seeds, onDelete, onDeleteAll }: UserSeedListProps) {
  function handleDeleteAll() {
    const confirmed = window.confirm(
      'Delete all your examples? This removes only user-created examples from this browser and cannot be undone.',
    );

    if (confirmed) {
      onDeleteAll();
    }
  }

  return (
    <section className="user-seed-list" aria-labelledby="user-seed-list-title">
      <div className="user-seed-list__header">
        <h2 id="user-seed-list-title" className="user-seed-list__title">
          Your examples ({seeds.length})
        </h2>
        {seeds.length > 0 && (
          <button
            type="button"
            className="user-seed-list__delete-all"
            onClick={handleDeleteAll}
          >
            Delete all my examples
          </button>
        )}
      </div>

      {seeds.length === 0 ? (
        <p className="user-seed-list__empty">
          You have not created any examples yet. Submit the form to add your first one.
        </p>
      ) : (
        <ul className="user-seed-list__items">
          {seeds.map((seed) => (
            <li key={seed.id}>
              <UserSeedCard seed={seed} onDelete={onDelete} />
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
