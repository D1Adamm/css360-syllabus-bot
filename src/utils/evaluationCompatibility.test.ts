import { describe, expect, it } from 'vitest';
import type { ComparisonRecord, EvaluationRecord } from '../types';
import {
  extractRecentComments,
  getRecentEvaluations,
  groupByQuestion,
  isEvaluationRecord,
  tallyCriterion,
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

/* ------------------------------------------------------------------------ *
 * The simplified evaluation form
 *
 * Helpfulness, concision and closeness to the syllabus were retired from the
 * student form. Records made before that keep all three and must keep
 * aggregating; records made after omit them and must not be dropped, counted as
 * a vote for anything, or shown as a zero.
 * ------------------------------------------------------------------------ */

/** A rating written by the simplified form: no retired criteria at all. */
const simplifiedRecord: EvaluationRecord = {
  id: 'evaluation-simplified',
  comparisonId: 'question-run-9',
  mostAccurate: 'fineTuned',
  preferredModel: 'fineTunedRag',
  hallucinationFlags: ['base', 'rag'],
  comment: 'The fine-tuned answer matched the syllabus wording.',
  createdAt: '2026-03-01T00:00:00.000Z',
  runId: 'run-9',
  questionText: 'When is the midterm?',
  courseId: 'css-350-spring-2026-n3h9',
};

describe('simplified evaluation records', () => {
  it('accepts a record that omits the retired criteria', () => {
    expect(isEvaluationRecord(simplifiedRecord)).toBe(true);
  });

  it('accepts the nulls the API sends for fields that were never set', () => {
    // An unset optional column is serialized as JSON null, not omitted.
    const fromApi = {
      ...simplifiedRecord,
      mostHelpful: null,
      mostConcise: null,
      bestGrounded: null,
      comment: null,
    };
    expect(isEvaluationRecord(fromApi)).toBe(true);
  });

  it('still rejects a record whose retired criterion is not an approach', () => {
    const broken = { ...simplifiedRecord, bestGrounded: 'gpt4' };
    expect(isEvaluationRecord(broken)).toBe(false);
  });

  it('still requires the two criteria the form asks for', () => {
    const { mostAccurate: _accurate, ...noAccuracy } = simplifiedRecord;
    const { preferredModel: _preferred, ...noPreference } = simplifiedRecord;

    expect(isEvaluationRecord(noAccuracy)).toBe(false);
    expect(isEvaluationRecord(noPreference)).toBe(false);
  });

  it('counts a retired criterion only among the records that answered it', () => {
    const tally = tallyCriterion([legacyRecord, liveRecord, simplifiedRecord], 'bestGrounded');

    expect(tally.answered).toBe(2);
    expect(tally.counts).toEqual({ base: 0, rag: 1, fineTuned: 0, fineTunedRag: 1 });
  });

  it('counts a retained criterion across every record, old and new', () => {
    const tally = tallyCriterion([legacyRecord, liveRecord, simplifiedRecord], 'preferredModel');

    expect(tally.answered).toBe(3);
    expect(tally.counts).toEqual({ base: 0, rag: 1, fineTuned: 0, fineTunedRag: 2 });
  });

  it('reports no answers at all when every record is a new one', () => {
    const tally = tallyCriterion([simplifiedRecord], 'mostHelpful');

    expect(tally.answered).toBe(0);
    expect(tally.counts).toEqual({ base: 0, rag: 0, fineTuned: 0, fineTunedRag: 0 });
  });

  it('leaves the grounding result out of a question no record answered it for', () => {
    const [result] = groupByQuestion([simplifiedRecord], comparisons);

    expect(result.mostGrounded).toBeNull();
    expect(result.groundedResponseCount).toBe(0);
    // The criteria that are still asked are unaffected.
    expect(result.mostPreferred).toBe('Fine-Tuned + RAG');
    expect(result.mostAccurate).toBe('Fine-Tuned');
  });

  it('still reports the grounding result for a question older records answered', () => {
    const [result] = groupByQuestion([legacyRecord], comparisons);

    expect(result.mostGrounded).toBe('RAG');
    expect(result.groundedResponseCount).toBe(1);
  });

  it('aggregates a mixed course without dropping either shape', () => {
    const results = groupByQuestion(
      [legacyRecord, liveRecord, simplifiedRecord],
      comparisons,
    );

    expect(results).toHaveLength(3);
    expect(results.reduce((sum, item) => sum + item.evaluationCount, 0)).toBe(3);
  });

  it('reads comments from old and new records alike', () => {
    const comments = extractRecentComments(
      [legacyRecord, liveRecord, simplifiedRecord],
      comparisons,
    );

    expect(comments.map((item) => item.question)).toEqual([
      'When is the midterm?',
      'Can I email about my grade?',
      'What is the late policy?',
    ]);
  });

  it('lists a new record in recent evaluations with both criteria it carries', () => {
    const recent = getRecentEvaluations([simplifiedRecord], comparisons);

    expect(recent[0].preferredModel).toBe('Fine-Tuned + RAG');
    expect(recent[0].mostAccurate).toBe('Fine-Tuned');
  });
});
