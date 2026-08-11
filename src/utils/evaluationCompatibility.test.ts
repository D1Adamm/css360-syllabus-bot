import { describe, expect, it } from 'vitest';
import type { ComparisonRecord, EvaluationRecord } from '../types';
import {
  extractRecentComments,
  getRecentEvaluations,
  groupByQuestion,
  isEvaluationRecord,
} from './evaluationUtils';

/**
 * Ratings written before live evaluation existed carry only `comparisonId`.
 * They must keep aggregating exactly as they always did, alongside new records
 * that also carry `runId` and `questionText`.
 */

const comparisons = [
  {
    id: 'comparison-1',
    question: 'What is the late policy?',
    category: 'Policies',
    relevantSyllabusSection: 'Late work',
    baseResponse: { text: '', grounding: 'Low', simulated: false },
    ragResponse: { text: '', grounding: 'High', simulated: false },
    fineTunedResponse: { text: '', grounding: 'Medium', simulated: false },
    fineTunedRagResponse: { text: '', grounding: 'High', simulated: false },
    notes: '',
  },
] as ComparisonRecord[];

const legacyRecord: EvaluationRecord = {
  id: 'evaluation-legacy',
  comparisonId: 'comparison-1',
  mostAccurate: 'rag',
  mostHelpful: 'rag',
  mostConcise: 'base',
  bestGrounded: 'rag',
  preferredModel: 'rag',
  hallucinationFlags: [],
  comment: 'Older note',
  createdAt: '2026-01-01T00:00:00.000Z',
};

const liveRecord: EvaluationRecord = {
  id: 'evaluation-live',
  comparisonId: 'question-run-7',
  mostAccurate: 'fineTunedRag',
  mostHelpful: 'fineTunedRag',
  mostConcise: 'base',
  bestGrounded: 'fineTunedRag',
  preferredModel: 'fineTunedRag',
  hallucinationFlags: ['base'],
  comment: 'Newer note',
  createdAt: '2026-02-01T00:00:00.000Z',
  runId: 'run-7',
  questionText: 'Can I email about my grade?',
  courseId: 'css-360-winter-2026-a7rp',
};

describe('evaluation record compatibility', () => {
  it('still validates a record written before the new optional fields existed', () => {
    expect(isEvaluationRecord(legacyRecord)).toBe(true);
  });

  it('validates a record carrying the new optional fields', () => {
    expect(isEvaluationRecord(liveRecord)).toBe(true);
  });

  it('groups a legacy record by its predefined question exactly as before', () => {
    const results = groupByQuestion([legacyRecord], comparisons);
    const group = results.find((item) => item.comparisonId === 'comparison-1');

    expect(group?.question).toBe('What is the late policy?');
    expect(group?.category).toBe('Policies');
    expect(group?.evaluationCount).toBe(1);
  });

  it('groups a free-text rating using the wording stored with it', () => {
    const results = groupByQuestion([liveRecord], comparisons);
    const group = results.find((item) => item.comparisonId === 'question-run-7');

    expect(group?.question).toBe('Can I email about my grade?');
    expect(group?.category).toBe('Student question');
  });

  it('aggregates legacy and live records together without dropping either', () => {
    const results = groupByQuestion([legacyRecord, liveRecord], comparisons);

    expect(results).toHaveLength(2);
    expect(results.map((item) => item.question).sort()).toEqual([
      'Can I email about my grade?',
      'What is the late policy?',
    ]);
  });

  it('shows readable question text in recent comments for both shapes', () => {
    const comments = extractRecentComments([legacyRecord, liveRecord], comparisons);

    expect(comments.map((item) => item.question)).toEqual([
      'Can I email about my grade?',
      'What is the late policy?',
    ]);
  });

  it('shows readable question text in recent evaluations for both shapes', () => {
    const recent = getRecentEvaluations([legacyRecord, liveRecord], comparisons);

    expect(recent.map((item) => item.question)).toEqual([
      'Can I email about my grade?',
      'What is the late policy?',
    ]);
  });

  it('falls back to the id when a record has neither a record nor stored wording', () => {
    const orphan: EvaluationRecord = { ...legacyRecord, comparisonId: 'gone' };
    const results = groupByQuestion([orphan], comparisons);

    expect(results[0]?.question).toBe('gone');
    expect(results[0]?.category).toBe('Unknown');
  });
});
