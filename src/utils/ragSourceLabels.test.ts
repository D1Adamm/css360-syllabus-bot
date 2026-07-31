/** @vitest-environment jsdom */
import { describe, expect, it } from 'vitest';
import type { RagGenerateSource } from '../lib/api';
import {
  cleanLabelFragment,
  formatRagSourceLabels,
  isGenericSectionTitle,
} from './ragSourceLabels';

function source(
  overrides: Partial<RagGenerateSource> & Pick<RagGenerateSource, 'chunkId' | 'sectionTitle' | 'text'>,
): RagGenerateSource {
  return {
    score: 0.9,
    ...overrides,
  };
}

describe('ragSourceLabels', () => {
  it('treats document-style titles as generic', () => {
    expect(isGenericSectionTitle('Software Engineering (Fall 2025)')).toBe(true);
    expect(isGenericSectionTitle('Late Policy')).toBe(false);
  });

  it('keeps unique meaningful section titles unchanged', () => {
    const labels = formatRagSourceLabels([
      source({
        chunkId: 'late-1',
        sectionTitle: 'Late Policy',
        text: 'Late work loses ten percent per day.',
      }),
      source({
        chunkId: 'office-1',
        sectionTitle: 'Office Hours',
        text: 'Office hours are Tuesdays at 2pm.',
      }),
    ]);

    expect(labels).toEqual(['Late Policy', 'Office Hours']);
  });

  it('distinguishes repeated generic titles with heading or preview text', () => {
    const labels = formatRagSourceLabels([
      source({
        chunkId: 'chunk-001',
        sectionTitle: 'Software Engineering (Fall 2025)',
        text: 'Grade Questions\nDiscuss grades privately with the instructor.',
      }),
      source({
        chunkId: 'chunk-002',
        sectionTitle: 'Software Engineering (Fall 2025)',
        text: 'Late Policy\nLate submissions lose credit each day without an extension.',
      }),
      source({
        chunkId: 'chunk-003',
        sectionTitle: 'Software Engineering (Fall 2025)',
        text: 'Software Engineering (Fall 2025)\nMakeup work is unavailable for missed labs except documented emergencies.',
      }),
    ]);

    expect(labels[0]).toBe('Grade Questions');
    expect(labels[1]).toBe('Late Policy');
    expect(labels[2]).toContain('Makeup work');
    expect(new Set(labels).size).toBe(3);
  });

  it('adds a preview when the same meaningful title repeats', () => {
    const labels = formatRagSourceLabels([
      source({
        chunkId: 'late-1',
        sectionTitle: 'Late Policy',
        text: 'Late project tasks lose half credit after twenty four hours.',
      }),
      source({
        chunkId: 'late-2',
        sectionTitle: 'Late Policy',
        text: 'One extension token may be used once per quarter for bot tasks.',
      }),
    ]);

    expect(labels[0]).toContain('Late Policy');
    expect(labels[1]).toContain('Late Policy');
    expect(labels[0]).not.toBe(labels[1]);
  });

  it('does not begin labels with and/or/punctuation when a cleaner sentence exists', () => {
    expect(cleanLabelFragment('and point it out to me when it happens.')).toBe(
      'Point it out to me when it happens.',
    );
    expect(cleanLabelFragment(', or email the instructor for help.')).toBe(
      'Email the instructor for help.',
    );

    const labels = formatRagSourceLabels([
      source({
        chunkId: 'chunk-010',
        sectionTitle: 'Software Engineering (Fall 2025)',
        text: 'and point it out to me when it happens. Students should wait one week.',
      }),
      source({
        chunkId: 'chunk-011',
        sectionTitle: 'Software Engineering (Fall 2025)',
        text: ', or email the instructor after checking Canvas announcements carefully.',
      }),
    ]);

    for (const label of labels) {
      expect(label.toLowerCase().startsWith('and ')).toBe(false);
      expect(label.toLowerCase().startsWith('or ')).toBe(false);
      expect(/^[,\-–—.]/.test(label)).toBe(false);
    }
    expect(labels[0]).toMatch(/^Point it out/i);
    expect(labels[1]).toMatch(/^Email the instructor/i);
    expect(new Set(labels).size).toBe(2);
  });

  it('disambiguates duplicate-looking source labels', () => {
    const labels = formatRagSourceLabels([
      source({
        chunkId: 'chunk-021',
        sectionTitle: 'Software Engineering (Fall 2025)',
        text: 'Students must contact the instructor before requesting any extension.',
      }),
      source({
        chunkId: 'chunk-022',
        sectionTitle: 'Software Engineering (Fall 2025)',
        text: 'Students must contact the instructor before requesting any extension.',
      }),
    ]);

    expect(labels[0]).not.toBe(labels[1]);
    expect(labels.some((label) => label.includes('chunk-021') || label.includes('chunk-022'))).toBe(
      true,
    );
  });
});
