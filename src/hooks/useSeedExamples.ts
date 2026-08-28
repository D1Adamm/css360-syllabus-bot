import { useCallback, useEffect, useState } from 'react';
import { useCourseId } from '../context/CourseContext';
import {
  createSeedExample,
  deleteAllUserSeedExamples,
  deleteSeedExample,
  subscribeToSeedExamples,
} from '../lib/seedExamplesDb';
import type { SeedExample } from '../types';

interface UseSeedExamplesResult {
  seeds: SeedExample[];
  loading: boolean;
  error: string | null;
  saving: boolean;
  saveError: string | null;
  addSeed: (seed: SeedExample) => Promise<void>;
  deleteSeed: (id: string) => Promise<void>;
  deleteAllSeeds: () => Promise<void>;
  clearSaveError: () => void;
}

/** Course-scoped seed examples from the `seed_examples` table. */
export function useSeedExamples(): UseSeedExamplesResult {
  const courseId = useCourseId();
  const [seeds, setSeeds] = useState<SeedExample[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setSeeds([]);
    setError(null);

    const unsubscribe = subscribeToSeedExamples(
      courseId,
      (nextSeeds) => {
        setSeeds(nextSeeds);
        setError(null);
        setLoading(false);
      },
      (message) => {
        setError(message);
        setLoading(false);
      },
    );

    return unsubscribe;
  }, [courseId]);

  const addSeed = useCallback(
    async (seed: SeedExample) => {
      setSaving(true);
      setSaveError(null);

      try {
        await createSeedExample(courseId, seed);
      } catch (caughtError) {
        const message =
          caughtError instanceof Error
            ? caughtError.message
            : 'Could not save the example.';
        setSaveError(message);
        throw caughtError;
      } finally {
        setSaving(false);
      }
    },
    [courseId],
  );

  const deleteSeed = useCallback(
    async (id: string) => {
      setSaveError(null);

      try {
        await deleteSeedExample(courseId, id);
      } catch (caughtError) {
        const message =
          caughtError instanceof Error
            ? caughtError.message
            : 'Could not delete the example.';
        setSaveError(message);
        throw caughtError;
      }
    },
    [courseId],
  );

  const deleteAllSeeds = useCallback(async () => {
    setSaveError(null);

    try {
      await deleteAllUserSeedExamples(courseId, seeds);
    } catch (caughtError) {
      const message =
        caughtError instanceof Error
          ? caughtError.message
          : 'Could not delete all examples.';
      setSaveError(message);
      throw caughtError;
    }
  }, [courseId, seeds]);

  const clearSaveError = useCallback(() => {
    setSaveError(null);
  }, []);

  return {
    seeds,
    loading,
    error,
    saving,
    saveError,
    addSeed,
    deleteSeed,
    deleteAllSeeds,
    clearSaveError,
  };
}
