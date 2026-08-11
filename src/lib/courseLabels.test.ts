import { describe, expect, it } from 'vitest';
import { formatCourseCode, formatCourseHeading } from './courseLabels';

describe('formatCourseCode', () => {
  it('upper-cases a course code however it was typed', () => {
    expect(formatCourseCode('Css 360')).toBe('CSS 360');
    expect(formatCourseCode('css 360')).toBe('CSS 360');
    expect(formatCourseCode('CSS360')).toBe('CSS 360');
    expect(formatCourseCode('css-490')).toBe('CSS 490');
    expect(formatCourseCode('  cSs   350  ')).toBe('CSS 350');
  });

  it('keeps a trailing section letter', () => {
    expect(formatCourseCode('css 360a')).toBe('CSS 360A');
  });

  it('leaves a name that is not code-shaped alone', () => {
    // Blanket upper-casing would shout at every course named with a phrase.
    expect(formatCourseCode('Intro to Programming')).toBe('Intro to Programming');
    expect(formatCourseCode('Senior Capstone')).toBe('Senior Capstone');
    expect(formatCourseCode('Data Structures 2')).toBe('Data Structures 2');
  });

  it('handles missing values without throwing', () => {
    expect(formatCourseCode(undefined)).toBe('');
    expect(formatCourseCode(null)).toBe('');
    expect(formatCourseCode('   ')).toBe('');
  });
});

describe('formatCourseHeading', () => {
  it('joins the code and subject', () => {
    expect(formatCourseHeading('css 360', 'Software Engineering')).toBe(
      'CSS 360 — Software Engineering',
    );
  });

  it('falls back to whichever part exists', () => {
    expect(formatCourseHeading('css 360', '')).toBe('CSS 360');
    expect(formatCourseHeading('', 'Software Engineering')).toBe(
      'Software Engineering',
    );
    expect(formatCourseHeading(null, null)).toBe('Course');
  });
});
