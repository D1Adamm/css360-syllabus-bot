import { onValue, push, ref, remove, set, type Unsubscribe } from 'firebase/database';
import type { EvaluationRecord } from '../types';
import { isEvaluationRecord } from '../utils/evaluationUtils';
import { assertValidCourseId } from './courseId';
import { getCourseEvaluationPath, getCourseEvaluationsPath } from './coursePaths';
import { database } from './firebase';

/** Legacy global path used by the current UI. Not yet migrated under courses/. */
export const EVALUATIONS_PATH = 'evaluations';

export function getEvaluationsRef() {
  return ref(database, EVALUATIONS_PATH);
}

export function getEvaluationRef(id: string) {
  return ref(database, `${EVALUATIONS_PATH}/${id}`);
}

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

/**
 * Subscribe to evaluations.
 * - Global (legacy UI): subscribeToEvaluations(onData, onError)
 * - Course-aware: subscribeToEvaluations(courseId, onData, onError)
 */
export function subscribeToEvaluations(
  onData: (evaluations: EvaluationRecord[]) => void,
  onError: (message: string) => void,
): Unsubscribe;
export function subscribeToEvaluations(
  courseId: string,
  onData: (evaluations: EvaluationRecord[]) => void,
  onError: (message: string) => void,
): Unsubscribe;
export function subscribeToEvaluations(
  courseIdOrOnData: string | ((evaluations: EvaluationRecord[]) => void),
  onDataOrOnError:
    | ((evaluations: EvaluationRecord[]) => void)
    | ((message: string) => void),
  maybeOnError?: (message: string) => void,
): Unsubscribe {
  if (typeof courseIdOrOnData === 'string') {
    assertValidCourseId(courseIdOrOnData);
    const onData = onDataOrOnError as (evaluations: EvaluationRecord[]) => void;
    const onError = maybeOnError as (message: string) => void;
    const evaluationsRef = getCourseEvaluationsRef(courseIdOrOnData);

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

  const onData = courseIdOrOnData;
  const onError = onDataOrOnError as (message: string) => void;
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

/**
 * Create an evaluation.
 * - Global (legacy UI): createEvaluation(evaluation)
 * - Course-aware: createEvaluation(courseId, evaluation)
 */
export async function createEvaluation(
  evaluation: EvaluationRecord,
): Promise<EvaluationRecord>;
export async function createEvaluation(
  courseId: string,
  evaluation: EvaluationRecord,
): Promise<EvaluationRecord>;
export async function createEvaluation(
  courseIdOrEvaluation: string | EvaluationRecord,
  maybeEvaluation?: EvaluationRecord,
): Promise<EvaluationRecord> {
  if (typeof courseIdOrEvaluation === 'string') {
    assertValidCourseId(courseIdOrEvaluation);
    const evaluation = maybeEvaluation as EvaluationRecord;
    const evaluationRef = push(getCourseEvaluationsRef(courseIdOrEvaluation));
    const storedEvaluation: EvaluationRecord = {
      ...evaluation,
      id: evaluationRef.key ?? evaluation.id,
      createdAt: evaluation.createdAt ?? new Date().toISOString(),
    };

    await set(evaluationRef, storedEvaluation);
    return storedEvaluation;
  }

  const evaluation = courseIdOrEvaluation;
  const evaluationRef = push(getEvaluationsRef());
  const storedEvaluation: EvaluationRecord = {
    ...evaluation,
    id: evaluationRef.key ?? evaluation.id,
    createdAt: evaluation.createdAt ?? new Date().toISOString(),
  };

  await set(evaluationRef, storedEvaluation);
  return storedEvaluation;
}

/**
 * Delete an evaluation.
 * - Global (legacy UI): deleteEvaluation(evaluationId)
 * - Course-aware: deleteEvaluation(courseId, evaluationId)
 */
export async function deleteEvaluation(id: string): Promise<void>;
export async function deleteEvaluation(courseId: string, evaluationId: string): Promise<void>;
export async function deleteEvaluation(
  courseIdOrId: string,
  maybeEvaluationId?: string,
): Promise<void> {
  if (maybeEvaluationId !== undefined) {
    assertValidCourseId(courseIdOrId);
    await remove(getCourseEvaluationRef(courseIdOrId, maybeEvaluationId));
    return;
  }

  await remove(getEvaluationRef(courseIdOrId));
}

export async function deleteAllEvaluations(evaluations: EvaluationRecord[]): Promise<void> {
  await Promise.all(evaluations.map((evaluation) => deleteEvaluation(evaluation.id)));
}
