/** @vitest-environment jsdom */
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import type { CourseMetadata } from '../types';

const subscribeToCoursesMock = vi.hoisted(() => vi.fn());

vi.mock('../lib/firebase', () => ({
  app: {},
  database: { name: 'mock-db' },
}));

vi.mock('../lib/coursesDb', () => ({
  subscribeToCourses: subscribeToCoursesMock,
}));

import { CoursePickerPage } from './CoursePickerPage';

interface CourseListItem {
  courseId: string;
  metadata: CourseMetadata;
}

function renderCoursePicker() {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <Routes>
        <Route path="/" element={<CoursePickerPage />} />
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

describe('CoursePickerPage', () => {
  beforeEach(() => {
    subscribeToCoursesMock.mockReset();
  });

  afterEach(() => {
    cleanup();
  });

  it('shows a loading state before Firebase responds', () => {
    subscribeToCoursesMock.mockImplementation(() => () => undefined);

    renderCoursePicker();

    expect(screen.getByRole('heading', { name: 'Loading courses' })).toBeInTheDocument();
  });

  it('displays Firebase courses and links each Open Course button to the course home', async () => {
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

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'CSS 350' })).toBeInTheDocument();
    });

    expect(screen.getByText('Management Principles')).toBeInTheDocument();
    expect(screen.getByText('Operating Systems')).toBeInTheDocument();
    expect(screen.getAllByText('Indexed').length).toBeGreaterThan(0);

    const css350Card = screen.getByRole('heading', { name: 'CSS 350' }).closest('li');
    const css430Card = screen.getByRole('heading', { name: 'CSS 430' }).closest('li');
    expect(css350Card).not.toBeNull();
    expect(css430Card).not.toBeNull();

    expect(within(css350Card as HTMLElement).getByRole('link', { name: 'Open Course' })).toHaveAttribute(
      'href',
      '/course/css-350-spring-2026-abcd/home',
    );
    expect(within(css430Card as HTMLElement).getByRole('link', { name: 'Open Course' })).toHaveAttribute(
      'href',
      '/course/css-430-summer-2026-ibce/home',
    );

    const cards = screen.getAllByRole('listitem');
    expect(within(cards[0] as HTMLElement).getByRole('heading', { name: 'CSS 350' })).toBeInTheDocument();
    expect(within(cards[1] as HTMLElement).getByRole('heading', { name: 'CSS 430' })).toBeInTheDocument();
  });

  it('shows an empty state when no courses are available', async () => {
    subscribeToCoursesMock.mockImplementation((onData: (courses: CourseListItem[]) => void) => {
      onData([]);
      return () => undefined;
    });

    renderCoursePicker();

    expect(
      await screen.findByRole('heading', { name: 'No courses available' }),
    ).toBeInTheDocument();
  });

  it('shows a Firebase error state', async () => {
    subscribeToCoursesMock.mockImplementation(
      (_onData: (courses: CourseListItem[]) => void, onError?: (message: string) => void) => {
        onError?.('Permission denied');
        return () => undefined;
      },
    );

    renderCoursePicker();

    expect(await screen.findByRole('heading', { name: 'Firebase error' })).toBeInTheDocument();
    expect(screen.getByText('Permission denied')).toBeInTheDocument();
  });

  it('links Create Course to /create-course', async () => {
    subscribeToCoursesMock.mockImplementation((onData: (courses: CourseListItem[]) => void) => {
      onData([]);
      return () => undefined;
    });

    renderCoursePicker();

    expect(screen.getByRole('link', { name: 'Create Course' })).toHaveAttribute(
      'href',
      '/create-course',
    );
  });
});
