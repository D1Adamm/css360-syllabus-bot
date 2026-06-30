import { onValue, push, ref, remove, set, type Unsubscribe } from 'firebase/database';
import type { EvaluationRecord } from '../types';
import { isEvaluationRecord } from '../utils/evaluationUtils';
import { database } from './firebase';

export const EVALUATIONS_PATH = 'evaluations';

export function getEvaluationsRef() {
  return ref(database, EVALUATIONS_PATH);
}

export function getEvaluationRef(id: string) {
  return ref(database, `${EVALUATIONS_PATH}/${id}`);
}

export function parseEvaluationsFromSnapshot(data: unknown): EvaluationRecord[] {
  if (!data || typeof data !== 'object') {
    return [];
  }

  return Object.values(data)
    .filter(isEvaluationRecord)
    .sort((left, right) => right.createdAt.localeCompare(left.createdAt));
}

export function subscribeToEvaluations(
  onData: (evaluations: EvaluationRecord[]) => void,
  onError: (message: string) => void,
): Unsubscribe {
  const evaluationsRef = getEvaluationsRef();

  return onValue(
    evaluationsRef,
    (snapshot) => {
      onData(parseEvaluationsFromSnapshot(snapshot.val()));
    },
    (error) => {
      onError(error.message);
    },
  );
}

export async function createEvaluation(evaluation: EvaluationRecord): Promise<EvaluationRecord> {
  const evaluationRef = push(getEvaluationsRef());
  const storedEvaluation: EvaluationRecord = {
    ...evaluation,
    id: evaluationRef.key ?? evaluation.id,
    createdAt: evaluation.createdAt ?? new Date().toISOString(),
  };

  await set(evaluationRef, storedEvaluation);
  return storedEvaluation;
}

export async function deleteEvaluation(id: string): Promise<void> {
  await remove(getEvaluationRef(id));
}

export async function deleteAllEvaluations(evaluations: EvaluationRecord[]): Promise<void> {
  await Promise.all(evaluations.map((evaluation) => deleteEvaluation(evaluation.id)));
}
