import type { SeedExample } from '../types';
import { UserSeedCard } from './UserSeedCard';

interface UserSeedListProps {
  seeds: SeedExample[];
  onDelete: (id: string) => void | Promise<void>;
  onDeleteAll: () => void | Promise<void>;
}

export function UserSeedList({ seeds, onDelete, onDeleteAll }: UserSeedListProps) {
  const userSeedCount = seeds.filter((seed) => seed.origin === 'user').length;

  async function handleDeleteAll() {
    const confirmed = window.confirm(
      'Delete all your examples? This removes all shared user-created examples and cannot be undone. AI-generated starter seeds are left unchanged.',
    );

    if (confirmed) {
      await onDeleteAll();
    }
  }

  return (
    <section className="user-seed-list" aria-labelledby="user-seed-list-title">
      <div className="user-seed-list__header">
        <h2 id="user-seed-list-title" className="user-seed-list__title">
          Course examples ({seeds.length})
        </h2>
        {userSeedCount > 0 && (
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
          No seed examples for this course yet. Submit the form to add your first one, or
          generate AI starter seeds from the backend.
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
