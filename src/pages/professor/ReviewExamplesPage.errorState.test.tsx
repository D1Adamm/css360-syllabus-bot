/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

const listCourseSeeds = vi.fn();

vi.mock('../../lib/api', () => ({
  ApiError: class ApiError extends Error {
    status?: number;
    constructor(message: string, status?: number) {
      super(message);
      this.name = 'ApiError';
      this.status = status;
    }
  },
  listCourseSeeds: (...args: unknown[]) => listCourseSeeds(...args),
  reviewCourseSeed: vi.fn(),
}));

vi.mock('../../context/CourseContext', () => ({
  useCourseId: () => 'css-360-winter-2026-a7rp',
}));

import { ReviewExamplesPage } from './ReviewExamplesPage';

/**
 * A failed load must not be dressed up as an empty course.
 *
 * The page used to render an error banner and, directly underneath it, filter
 * tabs reading 0 and "Nothing waiting for you". Zeroes look like fact; the
 * banner looks like noise. A professor could reasonably conclude their
 * students had contributed nothing.
 */
describe('ReviewExamplesPage load failure', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it('shows an error with retry and no zeroed counts or empty-state copy', async () => {
    listCourseSeeds.mockRejectedValue(new Error('boom'));

    render(<ReviewExamplesPage />);

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('Examples unavailable');

    // None of the "there is no data" surface may render.
    expect(screen.queryByRole('tab', { name: /Awaiting review/ })).toBeNull();
    expect(screen.queryByText('Nothing waiting for you')).toBeNull();
    expect(
      screen.queryByText(/No example questions have been collected/),
    ).toBeNull();
    expect(screen.queryByText(/^\d+ of \d+$/)).toBeNull();
  });

  it('retries the request when asked', async () => {
    listCourseSeeds.mockRejectedValue(new Error('boom'));

    render(<ReviewExamplesPage />);
    await screen.findByRole('alert');
    expect(listCourseSeeds).toHaveBeenCalledTimes(1);

    listCourseSeeds.mockResolvedValue({
      courseId: 'css-360-winter-2026-a7rp',
      count: 1,
      firebasePath: 'courses/css-360-winter-2026-a7rp/seedExamples',
      seeds: [
        {
          id: 'seed-1',
          question: 'Can I submit late?',
          answer: 'Late work may be submitted within 24 hours.',
          category: 'Late work',
          reviewStatus: 'generated',
          origin: 'ai_generated',
        },
      ],
    });

    fireEvent.click(screen.getByRole('button', { name: /Try again/ }));

    await waitFor(() => {
      expect(listCourseSeeds).toHaveBeenCalledTimes(2);
    });
    expect(await screen.findByText('Can I submit late?')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('distinguishes a genuinely empty course from a failure', async () => {
    listCourseSeeds.mockResolvedValue({
      courseId: 'css-360-winter-2026-a7rp',
      count: 0,
      firebasePath: 'courses/css-360-winter-2026-a7rp/seedExamples',
      seeds: [],
    });

    render(<ReviewExamplesPage />);

    expect(
      await screen.findByText(/No example questions have been collected/),
    ).toBeInTheDocument();
    expect(screen.queryByRole('alert')).toBeNull();
  });
});
