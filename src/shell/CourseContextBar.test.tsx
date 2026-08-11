/** @vitest-environment jsdom */
import { cleanup, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { CourseContextBar } from './CourseContextBar';

/**
 * The bar must not name a course it has not read yet.
 *
 * It used to render the literal word "Course" while the record loaded, so a
 * hard refresh flashed a course identity that was not the course being viewed.
 */
function renderBar(props: Partial<Parameters<typeof CourseContextBar>[0]> = {}) {
  return render(
    <MemoryRouter>
      <CourseContextBar items={[]} {...props} />
    </MemoryRouter>,
  );
}

afterEach(() => {
  cleanup();
});

describe('CourseContextBar', () => {
  it('shows a placeholder rather than a stand-in name while loading', () => {
    renderBar({ loading: true });

    expect(screen.getByTestId('course-identity-skeleton')).toBeInTheDocument();
    expect(screen.queryByText('Course')).not.toBeInTheDocument();
  });

  it('does not render a partially loaded identity', () => {
    // Title and term must not appear on their own before the name is known.
    renderBar({ loading: true, title: 'Software Engineering', term: 'Winter 2026' });

    expect(screen.queryByText('Software Engineering')).not.toBeInTheDocument();
    expect(screen.queryByText('Winter 2026')).not.toBeInTheDocument();
  });

  it('renders the real course identity once loaded', () => {
    renderBar({
      name: 'Css 360',
      title: 'Software Engineering',
      term: 'Winter 2026',
    });

    expect(screen.getByText('CSS 360')).toBeInTheDocument();
    expect(screen.getByText('Software Engineering')).toBeInTheDocument();
    expect(screen.queryByTestId('course-identity-skeleton')).not.toBeInTheDocument();
  });

  it('keeps the course links usable while the identity loads', () => {
    renderBar({
      loading: true,
      items: [{ to: '/student/course/abc/syllabus', label: 'Syllabus', icon: 'syllabus' }],
    });

    expect(screen.getByRole('link', { name: 'Syllabus' })).toBeInTheDocument();
  });
});
