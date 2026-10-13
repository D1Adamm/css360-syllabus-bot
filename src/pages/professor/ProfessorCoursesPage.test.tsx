/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import type { CourseMetadata } from '../../types';

const subscribeToCoursesMock = vi.hoisted(() => vi.fn());


vi.mock('../../lib/coursesDb', () => ({
  subscribeToCourses: subscribeToCoursesMock,
}));

import { SessionProvider, type Session } from '../../context/SessionContext';
import { mineOnlyStorageKey, ProfessorCoursesPage } from './ProfessorCoursesPage';

const ADMIN_SESSION: Session = {
  user: { userId: 'u-admin', email: 'a@uw.edu', displayName: 'A', role: 'admin', courseIds: [] },
  participants: [],
};

interface CourseListItem {
  courseId: string;
  metadata: CourseMetadata;
}

function renderCoursePicker(session: Session = ADMIN_SESSION) {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <SessionProvider initialSession={session}>
        <Routes>
          <Route path="/" element={<ProfessorCoursesPage />} />
          <Route path="/create-course" element={<div>Create course page</div>} />
          <Route path="/course/:courseId/home" element={<div>Course home</div>} />
        </Routes>
      </SessionProvider>
    </MemoryRouter>,
  );
}

function sampleCourse(
  courseId: string,
  overrides: Partial<CourseListItem['metadata']> = {},
): CourseListItem {
  return {
    courseId,
    metadata: {
      name: overrides.name ?? 'CSS 430',
      title: overrides.title ?? 'Operating Systems',
      term: overrides.term ?? 'Summer 2026',
      instructorName: overrides.instructorName ?? 'Ada Instructor',
      createdAt: overrides.createdAt ?? '2026-06-01T00:00:00.000Z',
      syllabusStatus: overrides.syllabusStatus ?? 'indexed',
      syllabusFileName: overrides.syllabusFileName ?? 'syllabus.pdf',
      syllabusType: overrides.syllabusType ?? 'pdf',
      chunkCount: overrides.chunkCount ?? 12,
    },
  };
}

describe('ProfessorCoursesPage', () => {
  beforeEach(() => {
    subscribeToCoursesMock.mockReset();
  });

  afterEach(() => {
    cleanup();
  });

  it('shows a loading state before the course list arrives', () => {
    subscribeToCoursesMock.mockImplementation(() => () => undefined);

    renderCoursePicker();

    expect(screen.getByText('Loading your courses…')).toBeInTheDocument();
  });

  it('lists courses newest first and links each into the professor course area', async () => {
    subscribeToCoursesMock.mockImplementation((onData: (courses: CourseListItem[]) => void) => {
      onData([
        sampleCourse('css-350-spring-2026-abcd', {
          name: 'CSS 350',
          title: 'Management Principles',
          createdAt: '2026-07-01T00:00:00.000Z',
        }),
        sampleCourse('css-430-summer-2026-ibce', {
          name: 'CSS 430',
          title: 'Operating Systems',
          createdAt: '2026-06-01T00:00:00.000Z',
        }),
      ]);
      return () => undefined;
    });

    renderCoursePicker();

    const css350 = await screen.findByRole('link', { name: /^CSS 350/ });
    expect(css350).toHaveAttribute('href', '/professor/course/css-350-spring-2026-abcd');
    expect(screen.getByRole('link', { name: /^CSS 430/ })).toHaveAttribute(
      'href',
      '/professor/course/css-430-summer-2026-ibce',
    );

    expect(screen.getByText('Management Principles')).toBeInTheDocument();
    expect(screen.getByText('Operating Systems')).toBeInTheDocument();
    expect(screen.getAllByText('Syllabus ready').length).toBe(2);

    const rows = screen.getAllByRole('listitem');
    expect(within(rows[0] as HTMLElement).getByRole('link', { name: /^CSS 350/ })).toBeInTheDocument();
    expect(within(rows[1] as HTMLElement).getByRole('link', { name: /^CSS 430/ })).toBeInTheDocument();
  });

  it('does not expose course ids or chunk counts to professors', async () => {
    subscribeToCoursesMock.mockImplementation((onData: (courses: CourseListItem[]) => void) => {
      onData([
        sampleCourse('css-350-spring-2026-abcd', {
          name: 'CSS 350',
          title: 'Management Principles',
        }),
      ]);
      return () => undefined;
    });

    renderCoursePicker();

    await screen.findByRole('link', { name: /^CSS 350/ });
    expect(screen.queryByText(/css-350-spring-2026-abcd/)).not.toBeInTheDocument();
    expect(screen.queryByText(/chunk/i)).not.toBeInTheDocument();
  });

  it('shows an empty state when no courses exist', async () => {
    subscribeToCoursesMock.mockImplementation((onData: (courses: CourseListItem[]) => void) => {
      onData([]);
      return () => undefined;
    });

    renderCoursePicker();

    expect(await screen.findByText('No courses yet')).toBeInTheDocument();
  });

  it('reports a load failure without naming the database', async () => {
    subscribeToCoursesMock.mockImplementation(
      (_onData: (courses: CourseListItem[]) => void, onError?: (message: string) => void) => {
        onError?.('PostgreSQL permission denied');
        return () => undefined;
      },
    );

    renderCoursePicker();

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('Courses unavailable');
    expect(alert).not.toHaveTextContent(/postgres/i);
  });

  it('links the create action into the professor area', async () => {
    subscribeToCoursesMock.mockImplementation((onData: (courses: CourseListItem[]) => void) => {
      onData([]);
      return () => undefined;
    });

    renderCoursePicker();

    const links = screen.getAllByRole('link', { name: 'Create course' });
    expect(links[0]).toHaveAttribute('href', '/professor/courses/new');
  });
  it('lists only courses the professor staffs, not one joined with a class code', async () => {
    subscribeToCoursesMock.mockImplementation((onData: (courses: CourseListItem[]) => void) => {
      onData([
        sampleCourse('css-430-summer-2026-ibce', { name: 'CSS 430' }),
        sampleCourse('css-350-spring-2026-abcd', { name: 'CSS 350' }),
      ]);
      return () => undefined;
    });

    renderCoursePicker({
      user: {
        userId: 'u-prof',
        email: 'p@uw.edu',
        displayName: 'P',
        role: 'professor',
        courseIds: ['css-430-summer-2026-ibce'],
      },
      participants: [{ courseId: 'css-350-spring-2026-abcd' }],
    });

    const list = await screen.findByRole('list', { name: 'Your courses' });
    expect(within(list).getAllByRole('link')).toHaveLength(1);
    expect(within(list).getByText('CSS 430')).toBeInTheDocument();
    expect(within(list).queryByText('CSS 350')).not.toBeInTheDocument();
  });
  it('lists a course the backend returned even before the cached session knows of it', async () => {
    subscribeToCoursesMock.mockImplementation((onData: (courses: CourseListItem[]) => void) => {
      onData([sampleCourse('css-430-summer-2026-ibce', { name: 'CSS 430' })]);
      return () => undefined;
    });

    // Granted by an administrator after this browser read its session.
    renderCoursePicker({
      user: { userId: 'u-prof', email: 'p@uw.edu', displayName: 'P', role: 'professor', courseIds: [] },
      participants: [],
    });

    const list = await screen.findByRole('list', { name: 'Your courses' });
    expect(within(list).getByText('CSS 430')).toBeInTheDocument();
  });

  describe('My courses only (administrators)', () => {
    const TEACHES = 'css-430-summer-2026-ibce';
    const OTHER = 'css-350-spring-2026-abcd';

    /** An instructor promoted to administrator: global access, one assignment kept. */
    const PROMOTED: Session = {
      user: { userId: 'u-promoted', email: 'p@uw.edu', displayName: 'P', role: 'admin', courseIds: [TEACHES] },
      participants: [],
    };

    beforeEach(() => {
      window.localStorage.clear();
      subscribeToCoursesMock.mockImplementation((onData: (courses: CourseListItem[]) => void) => {
        onData([
          sampleCourse(TEACHES, { name: 'CSS 430' }),
          sampleCourse(OTHER, { name: 'CSS 350' }),
        ]);
        return () => undefined;
      });
    });

    function names(): string[] {
      const list = screen.getByRole('list', { name: 'Your courses' });
      return within(list).getAllByRole('link').map((link) => link.textContent ?? '');
    }

    it('shows every course unchecked, and only assigned ones checked', async () => {
      renderCoursePicker(PROMOTED);
      const box = await screen.findByRole('checkbox', { name: 'My courses only' });
      expect(box).not.toBeChecked();
      await screen.findByRole('link', { name: /^CSS 350/ });
      expect(names()).toHaveLength(2);

      fireEvent.click(box);
      expect(box).toBeChecked();
      expect(names()).toHaveLength(1);
      expect(names()[0]).toMatch(/^CSS 430/);

      fireEvent.click(box);
      expect(names()).toHaveLength(2);
    });

    it('says so when nothing is assigned, rather than looking like there are no courses', async () => {
      renderCoursePicker(ADMIN_SESSION);
      fireEvent.click(await screen.findByRole('checkbox', { name: 'My courses only' }));
      expect(await screen.findByText('No courses are assigned to you')).toBeInTheDocument();
      expect(screen.queryByText('No courses yet')).not.toBeInTheDocument();
    });

    it('remembers the choice for this signed-in user only', async () => {
      renderCoursePicker(PROMOTED);
      fireEvent.click(await screen.findByRole('checkbox', { name: 'My courses only' }));
      expect(window.localStorage.getItem(mineOnlyStorageKey('u-promoted'))).toBe('1');
      cleanup();

      // The same person again: still narrowed.
      renderCoursePicker(PROMOTED);
      expect(await screen.findByRole('checkbox', { name: 'My courses only' })).toBeChecked();
      cleanup();

      // Another administrator on the same browser: their own default.
      renderCoursePicker(ADMIN_SESSION);
      expect(await screen.findByRole('checkbox', { name: 'My courses only' })).not.toBeChecked();
      expect(window.localStorage.getItem(mineOnlyStorageKey('u-admin'))).toBeNull();
    });

    it('is not offered to a professor, whose list is only ever their own courses', async () => {
      renderCoursePicker({
        user: { userId: 'u-prof', email: 'p@uw.edu', displayName: 'P', role: 'professor', courseIds: [TEACHES] },
        participants: [],
      });
      await screen.findByRole('link', { name: /^CSS 430/ });
      expect(screen.queryByRole('checkbox', { name: 'My courses only' })).toBeNull();
    });
  });
});
