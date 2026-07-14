import { describe, expect, it } from 'vitest';
import { DEFAULT_COURSE_ID } from './courseId';
import {
  coursePagePath,
  defaultCoursePagePath,
  getCourseIdFromPathname,
} from './courseRoutes';

describe('coursePagePath', () => {
  it('builds course-scoped page paths', () => {
    expect(coursePagePath('css360-default', 'home')).toBe('/course/css360-default/home');
    expect(coursePagePath('css360-default', 'compare')).toBe(
      '/course/css360-default/compare',
    );
    expect(coursePagePath('other-course', 'seeds')).toBe('/course/other-course/seeds');
  });

  it('preserves query strings', () => {
    expect(coursePagePath('css360-default', 'evaluate', '?comparison=1')).toBe(
      '/course/css360-default/evaluate?comparison=1',
    );
    expect(coursePagePath('css360-default', 'evaluate', 'comparison=1')).toBe(
      '/course/css360-default/evaluate?comparison=1',
    );
  });
});

describe('defaultCoursePagePath', () => {
  it('targets the default course id', () => {
    expect(defaultCoursePagePath('syllabus')).toBe(
      `/course/${DEFAULT_COURSE_ID}/syllabus`,
    );
  });
});

describe('getCourseIdFromPathname', () => {
  it('extracts a valid courseId from the pathname', () => {
    expect(getCourseIdFromPathname('/course/css360-default/compare')).toBe(
      'css360-default',
    );
    expect(getCourseIdFromPathname('/course/other-course/home')).toBe('other-course');
  });

  it('returns null for non-course or invalid course paths', () => {
    expect(getCourseIdFromPathname('/architecture')).toBeNull();
    expect(getCourseIdFromPathname('/compare')).toBeNull();
    expect(getCourseIdFromPathname('/course/Bad_Id/home')).toBeNull();
    expect(getCourseIdFromPathname('/course/-bad/home')).toBeNull();
  });
});
