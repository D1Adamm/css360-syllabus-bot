import { describe, expect, it } from 'vitest';
import { APPROACHES, approachDescription, approachLabel } from './approaches';
import { MODEL_KEYS } from '../../utils/evaluationUtils';

/**
 * The four approaches are named for the technique behind them, and every
 * user-facing surface reads those names from here. Pinning the copy keeps the
 * student, professor and admin views from drifting back into friendlier but
 * ambiguous names.
 */
describe('approach copy', () => {
  it('names each approach after its technical approach', () => {
    expect(APPROACHES.map((approach) => [approach.key, approach.label])).toEqual([
      ['base', 'Base'],
      ['rag', 'RAG'],
      ['fineTuned', 'Fine-Tuned'],
      ['fineTunedRag', 'Fine-Tuned + RAG'],
    ]);
  });

  it('describes each approach consistently with its name', () => {
    expect(APPROACHES.map((approach) => approach.description)).toEqual([
      'Base model, no course context',
      'Base model with retrieved syllabus context',
      'Course-specific fine-tuned model',
      'Fine-tuned model with retrieved syllabus context',
    ]);
  });

  it('uses none of the retired names', () => {
    const copy = APPROACHES.map(
      (approach) => `${approach.label} ${approach.description}`,
    ).join(' ');

    for (const retired of [/syllabus[- ]aware/i, /course[- ]trained/i]) {
      expect(copy).not.toMatch(retired);
    }
  });

  it('keeps the internal keys the backend and stored evaluations use', () => {
    expect(APPROACHES.map((approach) => approach.key)).toEqual(MODEL_KEYS);
  });

  it('looks up a label and a description by key', () => {
    expect(approachLabel('fineTunedRag')).toBe('Fine-Tuned + RAG');
    expect(approachDescription('rag')).toBe(
      'Base model with retrieved syllabus context',
    );
  });
});
