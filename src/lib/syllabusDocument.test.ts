import { describe, expect, it } from 'vitest';
import { parseSyllabusDocument } from './syllabusDocument';

describe('parseSyllabusDocument', () => {
  it('preserves paragraph breaks and line breaks within paragraphs', () => {
    const blocks = parseSyllabusDocument(
      [
        'Late Policy',
        '',
        'Work submitted after the deadline',
        'may receive reduced credit.',
        '',
        'Office Hours',
        '',
        'Tuesday at 2pm.',
      ].join('\n'),
    );

    expect(blocks).toEqual([
      { type: 'heading', text: 'Late Policy' },
      {
        type: 'paragraph',
        lines: ['Work submitted after the deadline', 'may receive reduced credit.'],
      },
      { type: 'heading', text: 'Office Hours' },
      { type: 'paragraph', lines: ['Tuesday at 2pm.'] },
    ]);
  });

  it('does not collapse multi-paragraph syllabus text into one block', () => {
    const blocks = parseSyllabusDocument('First paragraph.\n\nSecond paragraph.\n\nThird.');
    expect(blocks).toHaveLength(3);
    expect(blocks.every((block) => block.type === 'paragraph')).toBe(true);
  });
});
