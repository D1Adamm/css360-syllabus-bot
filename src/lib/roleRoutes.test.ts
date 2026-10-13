import { describe, expect, it } from 'vitest';
import { staffViewPath } from './roleRoutes';

const C = 'css360e-autumn-2026-c08m';

describe('staffViewPath', () => {
  it('keeps the course when switching from an admin course page to the professor view', () => {
    expect(staffViewPath('professor', `/admin/courses/${C}`)).toBe(`/professor/course/${C}`);
    expect(staffViewPath('professor', `/admin/courses/${C}/examples`)).toBe(`/professor/course/${C}`);
    expect(staffViewPath('professor', `/admin/courses/${C}/review`)).toBe(`/professor/course/${C}/examples`);
  });

  it('keeps the course when switching from a professor course page to the admin view', () => {
    expect(staffViewPath('admin', `/professor/course/${C}`)).toBe(`/admin/courses/${C}`);
    expect(staffViewPath('admin', `/professor/course/${C}/model`)).toBe(`/admin/courses/${C}`);
    expect(staffViewPath('admin', `/professor/course/${C}/examples`)).toBe(`/admin/courses/${C}/review`);
    expect(staffViewPath('admin', `/professor/course/${C}/ask`)).toBe(`/admin/courses/${C}`);
  });

  it('carries the course out of an admin preview of its student pages', () => {
    expect(staffViewPath('admin', `/student/course/${C}/compare`)).toBe(`/admin/courses/${C}`);
    expect(staffViewPath('professor', `/student/course/${C}/compare`)).toBe(`/professor/course/${C}`);
  });

  it('maps the course lists to each other and everything else to the landing page', () => {
    expect(staffViewPath('admin', '/professor/courses')).toBe('/admin/courses');
    expect(staffViewPath('admin', '/professor/courses/new')).toBe('/admin/courses');
    expect(staffViewPath('professor', '/admin/courses')).toBe('/professor/courses');
    expect(staffViewPath('professor', '/admin/people')).toBe('/professor/courses');
    expect(staffViewPath('admin', '/forbidden')).toBe('/admin');
  });

  it('never carries an invalid course id into the other view', () => {
    expect(staffViewPath('professor', '/admin/courses/Bad_Id')).toBe('/professor/courses');
    expect(staffViewPath('admin', '/professor/course/Bad_Id/model')).toBe('/admin');
  });
});
