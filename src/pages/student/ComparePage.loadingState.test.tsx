/** @vitest-environment jsdom */
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

/**
 * Nothing course-specific may be asserted before the course data resolves.
 *
 * On a hard refresh the page used to render the generic fallback suggestions
 * and a placeholder course name for a frame, then correct itself. Briefly
 * showing another course's questions under this course's header is a
 * correctness problem, not a cosmetic one — the reader has no way to know the
 * first render was a guess.
 */

const listCourseSeeds = vi.fn();

vi.mock('../../lib/api', () => ({
  ApiError: class ApiError extends Error {},
  generateBaseModel: vi.fn(),
  generateRag: vi.fn(),
  generateFineTuned: vi.fn(),
  generateFineTunedRag: vi.fn(),
  listCourseSeeds: (...args: unknown[]) => listCourseSeeds(...args),
}));

import { ComparisonRunProvider } from '../../context/ComparisonRunContext';
import { CourseRoute } from '../../components/CourseRoute';
import { GENERIC_SUGGESTIONS } from '../../hooks/useQuestionSuggestions';
import { ComparePage } from './ComparePage';

const COURSE_ID = 'css-360-winter-2026-a7rp';

function renderComparePage() {
  return render(
    <MemoryRouter initialEntries={[`/student/course/${COURSE_ID}/compare`]}>
      <ComparisonRunProvider>
        <Routes>
          <Route path="/student/course/:courseId" element={<CourseRoute />}>
            <Route path="compare" element={<ComparePage />} />
          </Route>
        </Routes>
      </ComparisonRunProvider>
    </MemoryRouter>,
  );
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

beforeEach(() => {
  vi.clearAllMocks();
  window.sessionStorage.clear();
});

afterEach(() => {
  cleanup();
});

describe('ComparePage suggestion loading state', () => {
  it('shows no suggestion chips at all on first render', () => {
    listCourseSeeds.mockReturnValue(new Promise(() => {}));

    renderComparePage();

    // Not the generic fallback, and not anything else.
    for (const question of GENERIC_SUGGESTIONS) {
      expect(screen.queryByRole('button', { name: question })).toBeNull();
    }
    expect(document.querySelectorAll('.ask__chip:not(.ask__chip--skeleton)')).toHaveLength(
      0,
    );
  });

  it('reserves the row with placeholders instead of leaving it empty', () => {
    listCourseSeeds.mockReturnValue(new Promise(() => {}));

    renderComparePage();

    // Placeholders keep the question box from jumping when chips arrive, and
    // are hidden from assistive technology.
    expect(document.querySelectorAll('.ask__chip--skeleton').length).toBeGreaterThan(0);
  });

  it('keeps the question box usable while suggestions load', () => {
    listCourseSeeds.mockReturnValue(new Promise(() => {}));

    renderComparePage();

    expect(
      screen.getByLabelText('What would you like to ask about this course?'),
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^Ask$/ })).toBeInTheDocument();
  });

  it('renders the course’s own questions once they resolve, and never before', async () => {
    const pending = deferred<{ seeds: unknown[] }>();
    listCourseSeeds.mockReturnValue(pending.promise);

    renderComparePage();

    expect(
      screen.queryByRole('button', { name: 'Is there a required textbook?' }),
    ).toBeNull();

    pending.resolve({
      seeds: [
        {
          id: '1',
          question: 'How do we commit to user stories during sprint planning?',
          answer: 'a',
          reviewStatus: 'approved',
        },
      ],
    });

    expect(
      await screen.findByRole('button', {
        name: 'How do we commit to user stories during sprint planning?',
      }),
    ).toBeInTheDocument();
    expect(screen.getByText('Questions from this course')).toBeInTheDocument();
    expect(document.querySelectorAll('.ask__chip--skeleton')).toHaveLength(0);
  });

  it('still falls back to generic questions for a course with none approved', async () => {
    listCourseSeeds.mockResolvedValue({ seeds: [] });

    renderComparePage();

    // The fallback is a resolved answer, so it appears only after the request
    // settles — never as an initial guess.
    expect(
      await screen.findByRole('button', { name: GENERIC_SUGGESTIONS[0] }),
    ).toBeInTheDocument();
    expect(screen.getByText('Try an example')).toBeInTheDocument();
  });

  it('falls back to generic questions when the request fails', async () => {
    listCourseSeeds.mockRejectedValue(new Error('offline'));

    renderComparePage();

    await waitFor(() => {
      expect(document.querySelectorAll('.ask__chip--skeleton')).toHaveLength(0);
    });
    expect(
      screen.getByRole('button', { name: GENERIC_SUGGESTIONS[0] }),
    ).toBeInTheDocument();
  });
});
