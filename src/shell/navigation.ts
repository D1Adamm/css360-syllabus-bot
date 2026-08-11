import type { IconName } from '../components/ui/icons';
import type { Role } from '../context/role';
import {
  professorCourseHomePath,
  professorCoursePath,
  studentCourseHomePath,
  studentCoursePath,
} from '../lib/roleRoutes';

export interface NavItem {
  to: string;
  label: string;
  icon: IconName;
  /** Match this path exactly rather than as a prefix. */
  end?: boolean;
}

/**
 * Primary navigation, defined once per role and rendered once per viewport.
 *
 * The old shell built an eleven-item list that mixed every role's concerns
 * together and rendered it twice — a desktop `<nav>` and a duplicate mobile
 * `<nav>`. Both problems are fixed here: each role gets at most five items,
 * and a single rendered list is restyled by CSS at narrow widths.
 */
export function primaryNavItems(role: Role, courseId: string | null): NavItem[] {
  if (role === 'student') {
    if (!courseId) {
      return [{ to: '/student', label: 'My courses', icon: 'course', end: true }];
    }
    return [
      { to: studentCourseHomePath(courseId), label: 'Home', icon: 'course', end: true },
      {
        to: studentCoursePath(courseId, 'contribute'),
        label: 'Contribute',
        icon: 'contribute',
      },
      { to: studentCoursePath(courseId, 'compare'), label: 'Compare', icon: 'compare' },
      {
        to: studentCoursePath(courseId, 'evaluate'),
        label: 'Evaluate',
        icon: 'evaluate',
      },
    ];
  }

  if (role === 'professor') {
    // Just Courses. The cross-course "Reviews" and "Models" hubs listed the
    // same courses again with one extra number each — a second way to reach
    // the same place, which made the active-nav state ambiguous and taught
    // nothing. Everything a professor does is scoped to one course, so the
    // course list is the only top-level destination. The routes still exist
    // and redirect, so old links keep working.
    return [{ to: '/professor/courses', label: 'Courses', icon: 'course' }];
  }

  return [
    { to: '/admin', label: 'Overview', icon: 'status', end: true },
    { to: '/admin/courses', label: 'Courses', icon: 'course' },
    { to: '/admin/training', label: 'Training', icon: 'upload' },
    { to: '/admin/models', label: 'Models', icon: 'model' },
    { to: '/admin/system', label: 'System', icon: 'admin' },
  ];
}

/** Compact secondary navigation shown inside a course. */
export function courseNavItems(role: Role, courseId: string): NavItem[] {
  if (role === 'student') {
    return [
      { to: studentCoursePath(courseId, 'syllabus'), label: 'Syllabus', icon: 'syllabus' },
    ];
  }

  if (role === 'professor') {
    return [
      // Exact match: the overview path is a prefix of every other course page.
      {
        to: professorCourseHomePath(courseId),
        label: 'Overview',
        icon: 'course',
        end: true,
      },
      {
        to: professorCoursePath(courseId, 'syllabus'),
        label: 'Syllabus',
        icon: 'syllabus',
      },
      {
        to: professorCoursePath(courseId, 'examples'),
        label: 'Examples',
        icon: 'review',
      },
      { to: professorCoursePath(courseId, 'model'), label: 'Model', icon: 'model' },
      {
        to: professorCoursePath(courseId, 'results'),
        label: 'Results',
        icon: 'evaluate',
      },
    ];
  }

  return [];
}
