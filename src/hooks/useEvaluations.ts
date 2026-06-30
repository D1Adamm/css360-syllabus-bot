import { useCallback, useEffect, useState } from 'react';
import {
  createEvaluation,
  deleteAllEvaluations,
  deleteEvaluation,
  subscribeToEvaluations,
} from '../lib/evaluationsDb';
import type { EvaluationRecord } from '../types';

interface UseEvaluationsResult {
  evaluations: EvaluationRecord[];
  loading: boolean;
  error: string | null;
  saving: boolean;
  saveError: string | null;
  addEvaluation: (evaluation: EvaluationRecord) => Promise<EvaluationRecord>;
  deleteEvaluation: (id: string) => Promise<void>;
  deleteAllEvaluations: () => Promise<void>;
  clearSaveError: () => void;
}

export function useEvaluations(): UseEvaluationsResult {
  const [evaluations, setEvaluations] = useState<EvaluationRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);

    const unsubscribe = subscribeToEvaluations(
      (nextEvaluations) => {
        setEvaluations(nextEvaluations);
        setError(null);
        setLoading(false);
      },
      (message) => {
        setError(message);
        setLoading(false);
      },
    );

    return unsubscribe;
  }, []);

  const addEvaluation = useCallback(async (evaluation: EvaluationRecord) => {
    setSaving(true);
    setSaveError(null);

    try {
      return await createEvaluation(evaluation);
    } catch (caughtError) {
      const message =
        caughtError instanceof Error
          ? caughtError.message
          : 'Could not save the evaluation to Firebase.';
      setSaveError(message);
      throw caughtError;
    } finally {
      setSaving(false);
    }
  }, []);

  const deleteEvaluationById = useCallback(async (id: string) => {
    setSaveError(null);

    try {
      await deleteEvaluation(id);
    } catch (caughtError) {
      const message =
        caughtError instanceof Error
          ? caughtError.message
          : 'Could not delete the evaluation from Firebase.';
      setSaveError(message);
      throw caughtError;
    }
  }, []);

  const deleteAllEvaluationsFromDb = useCallback(async () => {
    setSaveError(null);

    try {
      await deleteAllEvaluations(evaluations);
    } catch (caughtError) {
      const message =
        caughtError instanceof Error
          ? caughtError.message
          : 'Could not delete all evaluations from Firebase.';
      setSaveError(message);
      throw caughtError;
    }
  }, [evaluations]);

  const clearSaveError = useCallback(() => {
    setSaveError(null);
  }, []);

  return {
    evaluations,
    loading,
    error,
    saving,
    saveError,
    addEvaluation,
    deleteEvaluation: deleteEvaluationById,
    deleteAllEvaluations: deleteAllEvaluationsFromDb,
    clearSaveError,
  };
}
