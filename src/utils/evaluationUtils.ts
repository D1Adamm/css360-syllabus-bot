import { approachLabel } from '../components/compare/approaches';
import type { ComparisonRecord, EvaluationRecord, ModelKey } from '../types';

export const MODEL_KEYS: ModelKey[] = ['base', 'rag', 'fineTuned', 'fineTunedRag'];

/**
 * Human-readable names for the four approaches.
 *
 * Defined once in `components/compare/approaches.ts` and re-exported here so
 * evaluation and results screens cannot drift from the compare screen.
 */
export function getModelLabel(key: ModelKey): string {
  return approachLabel(key);
}

export function isValidModelKey(value: unknown): value is ModelKey {
  return typeof value === 'string' && MODEL_KEYS.includes(value as ModelKey);
}

/**
 * An optional field that was not set.
 *
 * `null` counts, not only `undefined`. The API serializes an unset optional
 * column as JSON `null` — `comment`, `runId`, `questionText` and now the
 * retired criteria all arrive that way — and a guard that accepted only
 * `undefined` rejected the whole record, which for `comment` meant every
 * rating a student submitted without a note was dropped before it reached the
 * results page.
 */
function isAbsent(value: unknown): boolean {
  return value === undefined || value === null;
}

/**
 * A criterion the form no longer asks for: valid when absent, and valid when
 * present with a real approach. Never valid as some other value — a record
 * carrying `mostHelpful: "gpt4"` is a record we cannot read, and dropping it is
 * still the right answer.
 */
function isOptionalModelKey(value: unknown): value is ModelKey | undefined {
  return isAbsent(value) || isValidModelKey(value);
}

export function isEvaluationRecord(value: unknown): value is EvaluationRecord {
  if (typeof value !== 'object' || value === null) {
    return false;
  }

  const record = value as EvaluationRecord;

  return (
    typeof record.id === 'string' &&
    typeof record.comparisonId === 'string' &&
    isValidModelKey(record.mostAccurate) &&
    isOptionalModelKey(record.mostHelpful) &&
    isOptionalModelKey(record.mostConcise) &&
    isOptionalModelKey(record.bestGrounded) &&
    isValidModelKey(record.preferredModel) &&
    Array.isArray(record.hallucinationFlags) &&
    record.hallucinationFlags.every(isValidModelKey) &&
    (isAbsent(record.comment) || typeof record.comment === 'string') &&
    typeof record.createdAt === 'string'
  );
}

export function isEvaluationRecordArray(value: unknown): value is EvaluationRecord[] {
  return Array.isArray(value) && value.every(isEvaluationRecord);
}

export function generateEvaluationId(): string {
  const timestamp = Date.now();
  const random = Math.random().toString(36).slice(2, 8);
  return `evaluation-${timestamp}-${random}`;
}

export function getTotalEvaluationCount(evaluations: EvaluationRecord[]): number {
  return evaluations.length;
}

export function getUniqueQuestionCount(evaluations: EvaluationRecord[]): number {
  return new Set(evaluations.map((evaluation) => evaluation.comparisonId)).size;
}

export interface ModelCounts {
  base: number;
  rag: number;
  fineTuned: number;
  fineTunedRag: number;
}

export function createEmptyModelCounts(): ModelCounts {
  return { base: 0, rag: 0, fineTuned: 0, fineTunedRag: 0 };
}

/**
 * A single-choice criterion on the evaluation form.
 *
 * `mostAccurate` and `preferredModel` are what every rating carries. The other
 * three were retired from the form and survive only on records made before
 * that, so any tally over them has to say how many records actually answered.
 */
export type EvaluationCriterion = keyof Pick<
  EvaluationRecord,
  'preferredModel' | 'mostAccurate' | 'mostHelpful' | 'mostConcise' | 'bestGrounded'
>;

export interface CriterionTally {
  counts: ModelCounts;
  /**
   * Records that answered this criterion — the denominator, and the only
   * honest one. Using the evaluation total instead would show a criterion that
   * three of forty records answered as though thirty-seven students had chosen
   * nothing.
   */
  answered: number;
}

export function tallyCriterion(
  evaluations: EvaluationRecord[],
  field: EvaluationCriterion,
): CriterionTally {
  const counts = createEmptyModelCounts();
  let answered = 0;

  for (const evaluation of evaluations) {
    const choice = evaluation[field];
    // A record that did not answer is skipped rather than counted anywhere.
    if (!isValidModelKey(choice)) {
      continue;
    }
    counts[choice] += 1;
    answered += 1;
  }

  return { counts, answered };
}

export function countByField(
  evaluations: EvaluationRecord[],
  field: EvaluationCriterion,
): ModelCounts {
  return tallyCriterion(evaluations, field).counts;
}

export function countHallucinationFlags(evaluations: EvaluationRecord[]): ModelCounts {
  const counts = createEmptyModelCounts();

  for (const evaluation of evaluations) {
    for (const flag of evaluation.hallucinationFlags) {
      counts[flag] += 1;
    }
  }

  return counts;
}

export function getTotalHallucinationFlags(evaluations: EvaluationRecord[]): number {
  return evaluations.reduce(
    (total, evaluation) => total + evaluation.hallucinationFlags.length,
    0,
  );
}

export function getMaxCount(counts: ModelCounts): number {
  return Math.max(...MODEL_KEYS.map((key) => counts[key]));
}

export function getTopModels(counts: ModelCounts): ModelKey[] {
  const max = getMaxCount(counts);
  if (max === 0) {
    return [];
  }
  return MODEL_KEYS.filter((key) => counts[key] === max);
}

export function formatTopModels(counts: ModelCounts): string {
  const topModels = getTopModels(counts);
  if (topModels.length === 0) {
    return 'No data';
  }
  const labels = topModels.map(getModelLabel);
  if (labels.length === 1) {
    return labels[0];
  }
  return `Tie: ${labels.join(' and ')}`;
}

export function calculatePercentage(count: number, total: number): number {
  if (total === 0) {
    return 0;
  }
  return Math.round((count / total) * 100);
}

export interface PerQuestionResult {
  comparisonId: string;
  question: string;
  category: string;
  evaluationCount: number;
  mostPreferred: string;
  mostAccurate: string;
  /**
   * Retired from the form. `null` for a question whose ratings were all made
   * after that, so a caller can leave it out rather than print "No data".
   */
  mostGrounded: string | null;
  /** Ratings that answered the retired grounding criterion. */
  groundedResponseCount: number;
  hallucinationFlags: ModelCounts;
}

export function groupByQuestion(
  evaluations: EvaluationRecord[],
  comparisons: ComparisonRecord[],
): PerQuestionResult[] {
  const comparisonMap = new Map(comparisons.map((record) => [record.id, record]));
  const grouped = new Map<string, EvaluationRecord[]>();

  for (const evaluation of evaluations) {
    const existing = grouped.get(evaluation.comparisonId) ?? [];
    existing.push(evaluation);
    grouped.set(evaluation.comparisonId, existing);
  }

  const results: PerQuestionResult[] = [];

  for (const [comparisonId, questionEvaluations] of grouped) {
    const comparison = comparisonMap.get(comparisonId);
    // Free-text questions have no predefined record; fall back to the wording
    // stored with the rating so they still group and read correctly.
    const storedQuestion = questionEvaluations.find((item) =>
      Boolean(item.questionText?.trim()),
    )?.questionText;
    const grounded = tallyCriterion(questionEvaluations, 'bestGrounded');
    results.push({
      comparisonId,
      question: comparison?.question ?? storedQuestion ?? comparisonId,
      category: comparison?.category ?? (storedQuestion ? 'Student question' : 'Unknown'),
      evaluationCount: questionEvaluations.length,
      mostPreferred: formatTopModels(countByField(questionEvaluations, 'preferredModel')),
      mostAccurate: formatTopModels(countByField(questionEvaluations, 'mostAccurate')),
      mostGrounded: grounded.answered > 0 ? formatTopModels(grounded.counts) : null,
      groundedResponseCount: grounded.answered,
      hallucinationFlags: countHallucinationFlags(questionEvaluations),
    });
  }

  return results.sort((left, right) => left.question.localeCompare(right.question));
}

export interface RecentComment {
  id: string;
  question: string;
  preferredModel: string;
  comment: string;
  createdAt: string;
}

export function extractRecentComments(
  evaluations: EvaluationRecord[],
  comparisons: ComparisonRecord[],
  limit = 5,
): RecentComment[] {
  const comparisonMap = new Map(comparisons.map((record) => [record.id, record]));

  return evaluations
    .filter((evaluation) => evaluation.comment && evaluation.comment.trim().length > 0)
    .sort((left, right) => right.createdAt.localeCompare(left.createdAt))
    .slice(0, limit)
    .map((evaluation) => ({
      id: evaluation.id,
      question:
        comparisonMap.get(evaluation.comparisonId)?.question ??
        evaluation.questionText ??
        evaluation.comparisonId,
      preferredModel: getModelLabel(evaluation.preferredModel),
      comment: evaluation.comment ?? '',
      createdAt: evaluation.createdAt,
    }));
}

export function getRecentEvaluations(
  evaluations: EvaluationRecord[],
  comparisons: ComparisonRecord[],
  limit = 5,
): {
  id: string;
  question: string;
  preferredModel: string;
  mostAccurate: string;
  createdAt: string;
}[] {
  const comparisonMap = new Map(comparisons.map((record) => [record.id, record]));

  return [...evaluations]
    .sort((left, right) => right.createdAt.localeCompare(left.createdAt))
    .slice(0, limit)
    .map((evaluation) => ({
      id: evaluation.id,
      question:
        comparisonMap.get(evaluation.comparisonId)?.question ??
        evaluation.questionText ??
        evaluation.comparisonId,
      preferredModel: getModelLabel(evaluation.preferredModel),
      mostAccurate: getModelLabel(evaluation.mostAccurate),
      createdAt: evaluation.createdAt,
    }));
}

export function formatEvaluationDate(isoDate: string): string {
  try {
    return new Date(isoDate).toLocaleString(undefined, {
      dateStyle: 'medium',
      timeStyle: 'short',
    });
  } catch {
    return isoDate;
  }
}

export function resolveComparisonId(
  paramId: string | null,
  comparisons: ComparisonRecord[],
): string {
  if (paramId && comparisons.some((record) => record.id === paramId)) {
    return paramId;
  }
  return comparisons[0]?.id ?? '';
}
