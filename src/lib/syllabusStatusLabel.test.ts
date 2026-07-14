import { describe, expect, it } from 'vitest';
import { formatSyllabusStatusLabel } from './syllabusStatusLabel';

describe('formatSyllabusStatusLabel', () => {
  it('maps syllabusStatus values to picker labels', () => {
    expect(formatSyllabusStatusLabel('indexed')).toBe('Indexed');
    expect(formatSyllabusStatusLabel('ready')).toBe('Indexed');
    expect(formatSyllabusStatusLabel('extracted')).toBe('Extracted');
    expect(formatSyllabusStatusLabel('uploaded')).toBe('Uploaded');
    expect(formatSyllabusStatusLabel('processing')).toBe('Uploaded');
    expect(formatSyllabusStatusLabel('not_uploaded')).toBe('Not uploaded');
    expect(formatSyllabusStatusLabel('none')).toBe('Not uploaded');
    expect(formatSyllabusStatusLabel('upload_failed')).toBe('Failed');
    expect(formatSyllabusStatusLabel('index_failed')).toBe('Failed');
    expect(formatSyllabusStatusLabel('error')).toBe('Failed');
  });
});
