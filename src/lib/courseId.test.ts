import { describe, expect, it } from 'vitest';
import {
  assertValidCourseId,
  DEFAULT_COURSE_ID,
  generateCourseId,
  isValidCourseId,
  slugifyCourseIdPart,
} from './courseId';
import {
  getCourseEvaluationPath,
  getCourseEvaluationsPath,
  getCourseMetadataPath,
  getCourseSeedExamplePath,
  getCourseSeedExamplesPath,
} from './coursePaths';

describe('isValidCourseId', () => {
  it('accepts valid course ids', () => {
    expect(isValidCourseId('css360')).toBe(true);
    expect(isValidCourseId('css360-default')).toBe(true);
    expect(isValidCourseId(DEFAULT_COURSE_ID)).toBe(true);
    expect(isValidCourseId('course-1')).toBe(true);
    expect(isValidCourseId('a')).toBe(true);
    expect(isValidCourseId('abc123')).toBe(true);
  });

  it('rejects empty, uppercase, and hyphen-edge ids', () => {
    expect(isValidCourseId('')).toBe(false);
    expect(isValidCourseId('CSS360')).toBe(false);
    expect(isValidCourseId('-css360')).toBe(false);
    expect(isValidCourseId('css360-')).toBe(false);
    expect(isValidCourseId('-')).toBe(false);
  });

  it('rejects path-unsafe characters and traversal patterns', () => {
    expect(isValidCourseId('course/id')).toBe(false);
    expect(isValidCourseId('course.id')).toBe(false);
    expect(isValidCourseId('course$id')).toBe(false);
    expect(isValidCourseId('course[id]')).toBe(false);
    expect(isValidCourseId('course]id')).toBe(false);
    expect(isValidCourseId('../evil')).toBe(false);
    expect(isValidCourseId('..')).toBe(false);
    expect(isValidCourseId('course\\id')).toBe(false);
    expect(isValidCourseId('a--b')).toBe(false);
  });

  it('rejects non-string values', () => {
    expect(isValidCourseId(null)).toBe(false);
    expect(isValidCourseId(undefined)).toBe(false);
    expect(isValidCourseId(123)).toBe(false);
  });
});

describe('assertValidCourseId', () => {
  it('throws for invalid course ids', () => {
    expect(() => assertValidCourseId('')).toThrow(/Invalid courseId/);
    expect(() => assertValidCourseId('../x')).toThrow(/Invalid courseId/);
    expect(() => assertValidCourseId('Bad_Id')).toThrow(/Invalid courseId/);
  });

  it('does not throw for valid course ids', () => {
    expect(() => assertValidCourseId('css360-default')).not.toThrow();
  });
});

describe('generateCourseId', () => {
  it('builds lowercase hyphenated ids from name and term', () => {
    const courseId = generateCourseId('CSS 430', 'Summer 2026');
    expect(courseId).toMatch(/^css-430-summer-2026-[a-z0-9]{4}$/);
    expect(isValidCourseId(courseId)).toBe(true);
  });

  it('removes unsafe characters from the input', () => {
    expect(slugifyCourseIdPart('CSS/430..OS$[Lab]!')).toBe('css-430-os-lab');
    const courseId = generateCourseId('CSS/430..OS$[Lab]!', 'Fall 2026');
    expect(courseId).toMatch(/^css-430-os-lab-fall-2026-[a-z0-9]{4}$/);
    expect(courseId).not.toMatch(/[./\\[\]$]/);
    expect(isValidCourseId(courseId)).toBe(true);
  });

  it('includes a short random suffix', () => {
    const first = generateCourseId('CSS 430', 'Summer 2026');
    expect(first).toMatch(/-[a-z0-9]{4}$/);
    expect(first.slice(-4)).toMatch(/^[a-z0-9]{4}$/);
  });

  it('does not begin or end with a hyphen', () => {
    const courseId = generateCourseId('  ---CSS 430---  ', '  ---Summer 2026---  ');
    expect(courseId.startsWith('-')).toBe(false);
    expect(courseId.endsWith('-')).toBe(false);
    expect(isValidCourseId(courseId)).toBe(true);
  });
});

describe('course Firebase paths', () => {
  const courseId = 'css360-default';

  it('builds the metadata path', () => {
    expect(getCourseMetadataPath(courseId)).toBe('courses/css360-default/metadata');
  });

  it('builds seed-example paths', () => {
    expect(getCourseSeedExamplesPath(courseId)).toBe(
      'courses/css360-default/seedExamples',
    );
    expect(getCourseSeedExamplePath(courseId, 'seed-1')).toBe(
      'courses/css360-default/seedExamples/seed-1',
    );
  });

  it('builds evaluation paths', () => {
    expect(getCourseEvaluationsPath(courseId)).toBe(
      'courses/css360-default/evaluations',
    );
    expect(getCourseEvaluationPath(courseId, 'eval-1')).toBe(
      'courses/css360-default/evaluations/eval-1',
    );
  });

  it('validates courseId before constructing paths', () => {
    expect(() => getCourseMetadataPath('../evil')).toThrow(/Invalid courseId/);
    expect(() => getCourseSeedExamplesPath('Bad')).toThrow(/Invalid courseId/);
    expect(() => getCourseEvaluationsPath('x/y')).toThrow(/Invalid courseId/);
  });
});
