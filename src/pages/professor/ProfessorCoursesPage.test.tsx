/** @vitest-environment jsdom */
import { cleanup, render, screen, within } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import type { CourseMetadata } from '../../types';

const subscribeToCoursesMock = vi.hoisted(() => vi.fn());


vi.mock('../../lib/coursesDb', () => ({
  subscribeToCourses: subscribeToCoursesMock,
}));

import { ProfessorCoursesPage } from './ProfessorCoursesPage';

interface CourseListItem {
  courseId: string;
  metadata: CourseMetadata;
}

function renderCoursePicker() {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <Routes>
        <Route path="/" element={<ProfessorCoursesPage />} />
        <Route path="/create-course" element={<div>Create course page</div>} />
        <Route path="/course/:courseId/home" element={<div>Course home</div>} />
      </Routes>
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
        onError?.('Firebase permission denied');
        return () => undefined;
      },
    );

    renderCoursePicker();

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('Courses unavailable');
    expect(alert).not.toHaveTextContent(/firebase/i);
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
});
