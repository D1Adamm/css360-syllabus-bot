import { useCallback, useMemo } from 'react';
import { PageHeader } from '../components/PageHeader';
import { SeedForm } from '../components/SeedForm';
import { UserSeedList } from '../components/UserSeedList';
import { useSeedExamples } from '../hooks/useSeedExamples';
import type { SeedExample } from '../types';

export function SeedBuilderPage() {
  const {
    seeds,
    loading,
    error,
    saving,
    saveError,
    addSeed,
    deleteSeed,
    deleteAllSeeds,
    clearSaveError,
  } = useSeedExamples();

  const manualSeeds = useMemo(
    () => seeds.filter((seed) => seed.origin === 'user'),
    [seeds],
  );

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
        description="Manually create question-and-answer examples for this course. AI-generated starter seeds are reviewed and browsed on Review Seeds and Dataset."
      />

      <aside className="seed-builder-notice" aria-label="Course storage notice">
        <p>
          <strong>Manual examples are stored under this course in Firebase Realtime Database.</strong>
        </p>
        <p>
          Seeds save to <code>courses/{'{courseId}'}/seedExamples</code> for the active course
          only. This page lists your manually created examples. Use Review Seeds and Dataset for
          AI-generated and reviewed seeds.
        </p>
      </aside>

      {error && (
        <p className="seed-builder-status seed-builder-status--error" role="alert">
          Could not load seed examples: {error}
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
            userSeeds={manualSeeds}
            onAddSeed={handleAddSeed}
            isSaving={saving}
            isLoading={loading}
          />
        </div>

        <div className="seed-builder-layout__list">
          {loading ? (
            <p className="seed-builder-status" role="status" aria-live="polite">
              Loading your seed examples…
            </p>
          ) : (
            <UserSeedList
              seeds={manualSeeds}
              onDelete={handleDeleteSeed}
              onDeleteAll={handleDeleteAll}
            />
          )}
        </div>
      </div>
    </>
  );
}
