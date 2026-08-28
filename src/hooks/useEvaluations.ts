import { useCallback, useEffect, useState } from 'react';
import { useCourseId } from '../context/CourseContext';
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

/** Course-scoped evaluations from the `evaluations` table. */
export function useEvaluations(): UseEvaluationsResult {
  const courseId = useCourseId();
  const [evaluations, setEvaluations] = useState<EvaluationRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setEvaluations([]);
    setError(null);

    const unsubscribe = subscribeToEvaluations(
      courseId,
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
  }, [courseId]);

  const addEvaluation = useCallback(
    async (evaluation: EvaluationRecord) => {
      setSaving(true);
      setSaveError(null);

      try {
        return await createEvaluation(courseId, evaluation);
      } catch (caughtError) {
        const message =
          caughtError instanceof Error
            ? caughtError.message
            : 'Could not save the evaluation.';
        setSaveError(message);
        throw caughtError;
      } finally {
        setSaving(false);
      }
    },
    [courseId],
  );

  const deleteEvaluationById = useCallback(
    async (id: string) => {
      setSaveError(null);

      try {
        await deleteEvaluation(courseId, id);
      } catch (caughtError) {
        const message =
          caughtError instanceof Error
            ? caughtError.message
            : 'Could not delete the evaluation.';
        setSaveError(message);
        throw caughtError;
      }
    },
    [courseId],
  );

  const deleteAllEvaluationsFromDb = useCallback(async () => {
    setSaveError(null);

    try {
      await deleteAllEvaluations(courseId, evaluations);
    } catch (caughtError) {
      const message =
        caughtError instanceof Error
          ? caughtError.message
          : 'Could not delete all evaluations.';
      setSaveError(message);
      throw caughtError;
    }
  }, [courseId, evaluations]);

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
