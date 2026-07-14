import { onValue, push, ref, remove, set, type Unsubscribe } from 'firebase/database';
import type { EvaluationRecord } from '../types';
import { isEvaluationRecord } from '../utils/evaluationUtils';
import { assertValidCourseId } from './courseId';
import { getCourseEvaluationPath, getCourseEvaluationsPath } from './coursePaths';
import { database } from './firebase';

export function getCourseEvaluationsRef(courseId: string) {
  return ref(database, getCourseEvaluationsPath(courseId));
}

export function getCourseEvaluationRef(courseId: string, evaluationId: string) {
  return ref(database, getCourseEvaluationPath(courseId, evaluationId));
}

export function parseEvaluationsFromSnapshot(data: unknown): EvaluationRecord[] {
  if (!data || typeof data !== 'object') {
    return [];
  }

  return Object.values(data)
    .filter(isEvaluationRecord)
    .sort((left, right) => right.createdAt.localeCompare(left.createdAt));
}

/** Subscribe to courses/{courseId}/evaluations. */
export function subscribeToEvaluations(
  courseId: string,
  onData: (evaluations: EvaluationRecord[]) => void,
  onError: (message: string) => void,
): Unsubscribe {
  assertValidCourseId(courseId);
  const evaluationsRef = getCourseEvaluationsRef(courseId);

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

/** Create an evaluation under courses/{courseId}/evaluations. */
export async function createEvaluation(
  courseId: string,
  evaluation: EvaluationRecord,
): Promise<EvaluationRecord> {
  assertValidCourseId(courseId);
  const evaluationRef = push(getCourseEvaluationsRef(courseId));
  const storedEvaluation: EvaluationRecord = {
    ...evaluation,
    id: evaluationRef.key ?? evaluation.id,
    createdAt: evaluation.createdAt ?? new Date().toISOString(),
  };

  await set(evaluationRef, storedEvaluation);
  return storedEvaluation;
}

/** Delete an evaluation from courses/{courseId}/evaluations/{evaluationId}. */
export async function deleteEvaluation(
  courseId: string,
  evaluationId: string,
): Promise<void> {
  assertValidCourseId(courseId);
  await remove(getCourseEvaluationRef(courseId, evaluationId));
}

export async function deleteAllEvaluations(
  courseId: string,
  evaluations: EvaluationRecord[],
): Promise<void> {
  await Promise.all(
    evaluations.map((evaluation) => deleteEvaluation(courseId, evaluation.id)),
  );
}
