import { useCallback } from 'react';
import { PageHeader } from '../components/PageHeader';
import { SeedForm } from '../components/SeedForm';
import { UserSeedList } from '../components/UserSeedList';
import { useLocalStorage } from '../hooks/useLocalStorage';
import type { SeedExample } from '../types';
import { USER_SEEDS_STORAGE_KEY, isSeedExampleArray } from '../utils/seedDataUtils';

export function SeedBuilderPage() {
  const [userSeeds, setUserSeeds] = useLocalStorage<SeedExample[]>(
    USER_SEEDS_STORAGE_KEY,
    [],
    isSeedExampleArray,
  );

  const handleAddSeed = useCallback(
    (seed: SeedExample) => {
      setUserSeeds((current) => [seed, ...current]);
    },
    [setUserSeeds],
  );

  const handleDeleteSeed = useCallback(
    (id: string) => {
      setUserSeeds((current) => current.filter((seed) => seed.id !== id));
    },
    [setUserSeeds],
  );

  const handleDeleteAll = useCallback(() => {
    setUserSeeds([]);
  }, [setUserSeeds]);

  return (
    <>
      <PageHeader
        title="Seed Data Builder"
        description="Create question-and-answer examples based on the syllabus. In a full course workflow, students would use this page to build training examples for fine-tuning experiments."
      />

      <aside className="seed-builder-notice" aria-label="Prototype storage notice">
        <p>
          <strong>In this prototype, your examples are stored only in this browser.</strong>
        </p>
        <p>
          Examples you create here are saved locally and are not automatically added to a
          trained model. They can be reviewed on the Seed Dataset page and exported as
          JSONL.
        </p>
      </aside>

      <div className="seed-builder-layout">
        <div className="seed-builder-layout__form">
          <h2 className="seed-builder-layout__section-title">Create a new example</h2>
          <SeedForm userSeeds={userSeeds} onAddSeed={handleAddSeed} />
        </div>

        <div className="seed-builder-layout__list">
          <UserSeedList
            seeds={userSeeds}
            onDelete={handleDeleteSeed}
            onDeleteAll={handleDeleteAll}
          />
        </div>
      </div>
    </>
  );
}
