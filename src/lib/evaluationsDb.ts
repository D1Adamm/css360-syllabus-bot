import type { EvaluationRecord } from '../types';
import { isEvaluationRecord } from '../utils/evaluationUtils';
import { assertValidCourseId } from './courseId';
import * as dbApi from './dbApi';
import { pollingSubscription, type Unsubscribe } from './pollingSubscription';

/**
 * Evaluations, now read and written through FastAPI against PostgreSQL.
 *
 * `isEvaluationRecord` still filters what arrives. Records written before free
 * text questions existed are missing fields the aggregation reads, and dropping
 * them here is what keeps the comparison charts from counting a partial record.
 */

export function parseEvaluationList(evaluations: unknown[]): EvaluationRecord[] {
  return evaluations
    .filter(isEvaluationRecord)
    .sort((left, right) => right.createdAt.localeCompare(left.createdAt));
}

export async function fetchEvaluations(courseId: string): Promise<EvaluationRecord[]> {
  assertValidCourseId(courseId);
  return parseEvaluationList((await dbApi.listEvaluations(courseId)).evaluations);
}

export function subscribeToEvaluations(
  courseId: string,
  onData: (evaluations: EvaluationRecord[]) => void,
  onError?: (message: string) => void,
): Unsubscribe {
  assertValidCourseId(courseId);

  return pollingSubscription<EvaluationRecord[]>({
    fetcher: () => fetchEvaluations(courseId),
    onData,
    onError,
  });
}

export async function createEvaluation(
  courseId: string,
  evaluation: EvaluationRecord,
): Promise<EvaluationRecord> {
  assertValidCourseId(courseId);

  return dbApi.createEvaluation(courseId, {
    ...evaluation,
    createdAt: evaluation.createdAt ?? new Date().toISOString(),
  });
}

export async function deleteEvaluation(
  courseId: string,
  evaluationId: string,
): Promise<void> {
  assertValidCourseId(courseId);
  await dbApi.deleteEvaluation(courseId, evaluationId);
}

/**
 * Clear every evaluation for one course.
 *
 * One request rather than a delete per record: the backend deletes by course
 * id, so a slow connection can no longer leave a course half-cleared.
 */
export async function deleteAllEvaluations(
  courseId: string,
  _evaluations?: EvaluationRecord[],
): Promise<void> {
  assertValidCourseId(courseId);
  await dbApi.deleteAllEvaluations(courseId);
}
