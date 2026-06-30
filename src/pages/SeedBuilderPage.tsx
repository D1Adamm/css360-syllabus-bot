import { useCallback } from 'react';
import { PageHeader } from '../components/PageHeader';
import { SeedForm } from '../components/SeedForm';
import { UserSeedList } from '../components/UserSeedList';
import { useSeedExamples } from '../hooks/useSeedExamples';
import type { SeedExample } from '../types';

export function SeedBuilderPage() {
  const {
    seeds: userSeeds,
    loading,
    error,
    saving,
    saveError,
    addSeed,
    deleteSeed,
    deleteAllSeeds,
    clearSaveError,
  } = useSeedExamples();

  const handleAddSeed = useCallback(
    async (seed: SeedExample) => {
      clearSaveError();
      await addSeed(seed);
    },
    [addSeed, clearSaveError],
  );

  const handleDeleteSeed = useCallback(
    async (id: string) => {
      clearSaveError();
      await deleteSeed(id);
    },
    [clearSaveError, deleteSeed],
  );

  const handleDeleteAll = useCallback(async () => {
    clearSaveError();
    await deleteAllSeeds();
  }, [clearSaveError, deleteAllSeeds]);

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

      {error && (
        <p className="seed-builder-status seed-builder-status--error" role="alert">
          Could not load shared seed examples: {error}
        </p>
      )}

      {saveError && (
        <p className="seed-builder-status seed-builder-status--error" role="alert">
          {saveError}
        </p>
      )}

      <div className="seed-builder-layout">
        <div className="seed-builder-layout__form">
          <h2 className="seed-builder-layout__section-title">Create a new example</h2>
          <SeedForm
            userSeeds={userSeeds}
            onAddSeed={handleAddSeed}
            isSaving={saving}
            isLoading={loading}
          />
        </div>

        <div className="seed-builder-layout__list">
          {loading ? (
            <p className="seed-builder-status" role="status" aria-live="polite">
              Loading shared seed examples…
            </p>
          ) : (
            <UserSeedList
              seeds={userSeeds}
              onDelete={handleDeleteSeed}
              onDeleteAll={handleDeleteAll}
            />
          )}
        </div>
      </div>
    </>
  );
}
