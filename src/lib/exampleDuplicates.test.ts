import { describe, expect, it } from 'vitest';
import { findDuplicateExampleIds, normalizeQuestionKey } from './exampleDuplicates';

describe('normalizeQuestionKey', () => {
  it('ignores case, punctuation and spacing', () => {
    expect(normalizeQuestionKey('Where  are office HOURS?')).toBe(
      'where are office hours',
    );
  });
});

describe('findDuplicateExampleIds', () => {
  it('flags every member of a repeated question, not just the later one', () => {
    const duplicates = findDuplicateExampleIds([
      { id: 'a', question: 'Where are office hours?' },
      { id: 'b', question: 'where are office hours' },
      { id: 'c', question: 'When is the final?' },
    ]);

    expect([...duplicates].sort()).toEqual(['a', 'b']);
  });

  it('prefers the key generation already stored', () => {
    const duplicates = findDuplicateExampleIds([
      { id: 'a', question: 'One wording', normalizedQuestionKey: 'shared key' },
      { id: 'b', question: 'Another wording', normalizedQuestionKey: 'shared key' },
    ]);

    expect(duplicates.size).toBe(2);
  });

  it('claims nothing about merely similar questions', () => {
    const duplicates = findDuplicateExampleIds([
      { id: 'a', question: 'Where are office hours?' },
      { id: 'b', question: 'What time are office hours?' },
    ]);

    expect(duplicates.size).toBe(0);
  });

  it('ignores records with no id or no question', () => {
    const duplicates = findDuplicateExampleIds([
      { id: '', question: 'Same question' },
      { id: 'b', question: 'Same question' },
      { id: 'c', question: '' },
      { id: 'd', question: '' },
    ]);

    expect(duplicates.size).toBe(0);
  });
});
