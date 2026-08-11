import { describe, expect, it } from 'vitest';
import type { CourseSeedReviewRecord } from './api';
import {
  countExamples,
  exampleAnswer,
  exampleQuestion,
  exampleWasEdited,
  resolveExampleStatus,
} from './exampleCounts';
import { getModelReadiness, RECOMMENDED_APPROVED_EXAMPLES } from './modelStatus';

function example(overrides: CourseSeedReviewRecord): CourseSeedReviewRecord {
  return overrides;
}

describe('resolveExampleStatus', () => {
  it('prefers reviewStatus over the older status field', () => {
    expect(
      resolveExampleStatus(example({ reviewStatus: 'approved', status: 'generated' })),
    ).toBe('approved');
  });

  it('falls back to status for records written before reviewStatus existed', () => {
    expect(resolveExampleStatus(example({ status: 'rejected' }))).toBe('rejected');
  });

  it('treats a record with neither field as awaiting review', () => {
    expect(resolveExampleStatus(example({ id: 'x' }))).toBe('generated');
  });
});

describe('countExamples', () => {
  it('counts each review state separately', () => {
    const counts = countExamples([
      example({ reviewStatus: 'approved' }),
      example({ reviewStatus: 'approved' }),
      example({ reviewStatus: 'rejected' }),
      example({ reviewStatus: 'edited' }),
      example({ reviewStatus: 'generated' }),
    ]);

    expect(counts).toEqual({
      total: 5,
      approved: 2,
      rejected: 1,
      edited: 1,
      pending: 1,
    });
  });

  it('counts an unrecognised status as pending rather than dropping it', () => {
    const counts = countExamples([
      example({ reviewStatus: 'something-old' }),
      example({ id: 'no-status' }),
    ]);

    expect(counts.pending).toBe(2);
    expect(counts.total).toBe(2);
  });

  it('returns zeroes for an empty course', () => {
    expect(countExamples([])).toEqual({
      total: 0,
      approved: 0,
      rejected: 0,
      edited: 0,
      pending: 0,
    });
  });
});

describe('exampleWasEdited', () => {
  it('is true when the flag is set', () => {
    expect(exampleWasEdited(example({ wasEdited: true }))).toBe(true);
  });

  it('is true for an edited status', () => {
    expect(exampleWasEdited(example({ reviewStatus: 'edited' }))).toBe(true);
  });

  it('stays true after a later approval, via preserved provenance', () => {
    expect(
      exampleWasEdited(
        example({ reviewStatus: 'approved', originalQuestion: 'The first wording' }),
      ),
    ).toBe(true);
  });

  it('is false for an untouched example', () => {
    expect(exampleWasEdited(example({ reviewStatus: 'generated' }))).toBe(false);
  });
});

describe('question and answer fallbacks', () => {
  it('reads the newer field names', () => {
    const seed = example({ question: 'Q', answer: 'A' });
    expect(exampleQuestion(seed)).toBe('Q');
    expect(exampleAnswer(seed)).toBe('A');
  });

  it('falls back to the older instruction and response fields', () => {
    const seed = example({ instruction: 'Old Q', response: 'Old A' });
    expect(exampleQuestion(seed)).toBe('Old Q');
    expect(exampleAnswer(seed)).toBe('Old A');
  });

  it('returns empty strings for an incomplete record', () => {
    expect(exampleQuestion(example({ id: 'x' }))).toBe('');
    expect(exampleAnswer(example({ id: 'x' }))).toBe('');
  });
});

describe('getModelReadiness', () => {
  it('reports how many more approved examples are needed', () => {
    const readiness = getModelReadiness(10);
    expect(readiness.hasEnough).toBe(false);
    expect(readiness.remaining).toBe(RECOMMENDED_APPROVED_EXAMPLES - 10);
  });

  it('is satisfied at the recommended count', () => {
    const readiness = getModelReadiness(RECOMMENDED_APPROVED_EXAMPLES);
    expect(readiness.hasEnough).toBe(true);
    expect(readiness.remaining).toBe(0);
  });

  it('never reports a negative remainder', () => {
    expect(getModelReadiness(500).remaining).toBe(0);
  });
});
