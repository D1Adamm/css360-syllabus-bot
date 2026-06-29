import type { ComparisonRecord, EvaluationRecord, ModelKey } from '../types';

export const EVALUATIONS_STORAGE_KEY = 'syllabus-demo-evaluations';

export const MODEL_KEYS: ModelKey[] = ['base', 'rag', 'fineTuned', 'fineTunedRag'];

const MODEL_LABELS: Record<ModelKey, string> = {
  base: 'Base Model',
  rag: 'RAG',
  fineTuned: 'Fine-Tuned Model',
  fineTunedRag: 'Fine-Tuned + RAG',
};

export function getModelLabel(key: ModelKey): string {
  return MODEL_LABELS[key];
}

export function isValidModelKey(value: unknown): value is ModelKey {
  return typeof value === 'string' && MODEL_KEYS.includes(value as ModelKey);
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
    isValidModelKey(record.mostHelpful) &&
    isValidModelKey(record.mostConcise) &&
    isValidModelKey(record.bestGrounded) &&
    isValidModelKey(record.preferredModel) &&
    Array.isArray(record.hallucinationFlags) &&
    record.hallucinationFlags.every(isValidModelKey) &&
    (record.comment === undefined || typeof record.comment === 'string') &&
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

export function countByField(
  evaluations: EvaluationRecord[],
  field: keyof Pick<
    EvaluationRecord,
    'preferredModel' | 'mostAccurate' | 'mostHelpful' | 'mostConcise' | 'bestGrounded'
  >,
): ModelCounts {
  const counts = createEmptyModelCounts();

  for (const evaluation of evaluations) {
    counts[evaluation[field]] += 1;
  }

  return counts;
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
  mostGrounded: string;
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
    results.push({
      comparisonId,
      question: comparison?.question ?? comparisonId,
      category: comparison?.category ?? 'Unknown',
      evaluationCount: questionEvaluations.length,
      mostPreferred: formatTopModels(countByField(questionEvaluations, 'preferredModel')),
      mostAccurate: formatTopModels(countByField(questionEvaluations, 'mostAccurate')),
      mostGrounded: formatTopModels(countByField(questionEvaluations, 'bestGrounded')),
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
      question: comparisonMap.get(evaluation.comparisonId)?.question ?? evaluation.comparisonId,
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
      question: comparisonMap.get(evaluation.comparisonId)?.question ?? evaluation.comparisonId,
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
