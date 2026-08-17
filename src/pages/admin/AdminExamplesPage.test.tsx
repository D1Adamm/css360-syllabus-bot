/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import type { SeedExample } from '../../types';
import { CourseProvider } from '../../context/CourseContext';
import { AdminExamplesPage } from './AdminExamplesPage';

const subscribeToSeedExamplesMock = vi.hoisted(() => vi.fn());


vi.mock('../../lib/seedExamplesDb', () => ({
  subscribeToSeedExamples: subscribeToSeedExamplesMock,
  createSeedExample: vi.fn(),
  deleteSeedExample: vi.fn(),
  deleteAllSeedExamples: vi.fn(),
  deleteAllUserSeedExamples: vi.fn(),
  updateSeedExample: vi.fn(),
}));

function makeSeed(
  courseLabel: string,
  id: string,
  options: {
    reviewStatus?: SeedExample['reviewStatus'];
    instruction?: string;
  } = {},
): SeedExample {
  const reviewStatus = options.reviewStatus ?? 'approved';
  return {
    id,
    instruction: options.instruction ?? `When does ${courseLabel} meet?`,
    response: `${courseLabel} meets on a course-specific schedule described in its syllabus.`,
    category: 'Course Basics',
    sourceSection: `${courseLabel} Meetings`,
    difficulty: 'Easy',
    directlyAnswered: true,
    origin: 'user',
    createdAt: '2026-07-01T00:00:00.000Z',
    reviewStatus,
    status: reviewStatus,
  };
}

function renderDataset(courseId: string) {
  return render(
    <MemoryRouter initialEntries={[`/course/${courseId}/dataset`]}>
      <Routes>
        <Route
          path="/course/:courseId/dataset"
          element={
            <CourseProvider courseId={courseId}>
              <AdminExamplesPage />
            </CourseProvider>
          }
        />
      </Routes>
    </MemoryRouter>,
  );
}

describe('AdminExamplesPage course separation', () => {
  beforeEach(() => {
    subscribeToSeedExamplesMock.mockReset();
  });

  afterEach(() => {
    cleanup();
  });

  it('loads only the route course seedExamples path via the courseId subscription', async () => {
    subscribeToSeedExamplesMock.mockImplementation(
      (courseId: string, onData: (seeds: SeedExample[]) => void) => {
        onData([makeSeed('CSS 430', `${courseId}-seed-1`)]);
        return () => undefined;
      },
    );

    renderDataset('css-430-summer-2026-ibce');

    await waitFor(() => {
      expect(subscribeToSeedExamplesMock).toHaveBeenCalledWith(
        'css-430-summer-2026-ibce',
        expect.any(Function),
        expect.any(Function),
      );
    });

    // Admin is the one place the storage path is acceptable; it now appears in
    // the page description rather than a notice.
    expect(screen.getByText(/css-430-summer-2026-ibce/)).toBeInTheDocument();
    expect(screen.getByText(/When does CSS 430 meet\?/)).toBeInTheDocument();
    expect(screen.queryByText(/CSS360|CSS 360/i)).not.toBeInTheDocument();
  });

  it('keeps CSS 350 and CSS 430 datasets fully separate', async () => {
    subscribeToSeedExamplesMock.mockImplementation(
      (courseId: string, onData: (seeds: SeedExample[]) => void) => {
        if (courseId === 'css-350-spring-2026-abcd') {
          onData([makeSeed('CSS 350', 'css350-only')]);
        } else {
          onData([makeSeed('CSS 430', 'css430-only')]);
        }
        return () => undefined;
      },
    );

    const first = renderDataset('css-350-spring-2026-abcd');
    expect(await screen.findByText(/When does CSS 350 meet\?/)).toBeInTheDocument();
    expect(screen.queryByText(/When does CSS 430 meet\?/)).not.toBeInTheDocument();
    first.unmount();

    renderDataset('css-430-summer-2026-ibce');
    expect(await screen.findByText(/When does CSS 430 meet\?/)).toBeInTheDocument();
    expect(screen.queryByText(/When does CSS 350 meet\?/)).not.toBeInTheDocument();
  });

  it('shows an empty state when the course has no seed examples', async () => {
    subscribeToSeedExamplesMock.mockImplementation(
      (_courseId: string, onData: (seeds: SeedExample[]) => void) => {
        onData([]);
        return () => undefined;
      },
    );

    renderDataset('css-430-summer-2026-ibce');

    expect(
      await screen.findByText('No examples stored'),
    ).toBeInTheDocument();
    expect(screen.queryByLabelText('Seed examples')).not.toBeInTheDocument();
  });

  it('does not import local CSS 360 prototype seed JSON into the dataset list', async () => {
    subscribeToSeedExamplesMock.mockImplementation(
      (_courseId: string, onData: (seeds: SeedExample[]) => void) => {
        onData([]);
        return () => undefined;
      },
    );

    renderDataset('css-430-summer-2026-ibce');

    await screen.findByText('No examples stored');

    // Phrase commonly present in the legacy CSS 360 prototype seed export.
    expect(screen.queryByText(/UW1 room 302/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/prototype examples/i)).not.toBeInTheDocument();
  });

  it('defaults to approved seeds and summarizes review status counts', async () => {
    subscribeToSeedExamplesMock.mockImplementation(
      (_courseId: string, onData: (seeds: SeedExample[]) => void) => {
        onData([
          makeSeed('CSS 360', 'approved-1', {
            reviewStatus: 'approved',
            instruction: 'Approved: when does CSS 360 meet?',
          }),
          makeSeed('CSS 360', 'generated-1', {
            reviewStatus: 'generated',
            instruction: 'Generated: what is the late work policy?',
          }),
          makeSeed('CSS 360', 'rejected-1', {
            reviewStatus: 'rejected',
            instruction: 'Rejected: can I skip standups?',
          }),
          makeSeed('CSS 360', 'edited-1', {
            reviewStatus: 'edited',
            instruction: 'Edited: how does grading work?',
          }),
        ]);
        return () => undefined;
      },
    );

    renderDataset('css-360-winter-2026-a7rp');

    expect(await screen.findByText('Awaiting review')).toBeInTheDocument();
    const stats = document.querySelector('.admin-stats');
    expect(stats).not.toBeNull();
    const cards = Array.from(stats!.querySelectorAll('.admin-stat')).map((card) => ({
      label: card.querySelector('dt')?.textContent,
      value: card.querySelector('dd')?.textContent,
    }));
    expect(cards).toEqual([
      { label: 'Total', value: '4' },
      { label: 'Approved', value: '1' },
      { label: 'Rejected', value: '1' },
      { label: 'Awaiting review', value: '1' },
      { label: 'Edited', value: '1' },
    ]);

    expect(screen.getByRole('tab', { name: 'Approved' })).toHaveAttribute(
      'aria-selected',
      'true',
    );
    expect(screen.getByText(/Approved: when does CSS 360 meet\?/)).toBeInTheDocument();
    expect(screen.queryByText(/Generated: what is the late work policy\?/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Rejected: can I skip standups\?/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Edited: how does grading work\?/)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('tab', { name: 'Generated' }));
    expect(screen.getByText(/Generated: what is the late work policy\?/)).toBeInTheDocument();
    expect(screen.queryByText(/Approved: when does CSS 360 meet\?/)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('tab', { name: 'All' }));
    expect(screen.getByText('Showing 4 of 4')).toBeInTheDocument();
    expect(screen.getByText(/Edited: how does grading work\?/)).toBeInTheDocument();

    const categoryFilter = screen.getByLabelText('Category');
    expect(categoryFilter.tagName).toBe('SELECT');
    expect(categoryFilter).toHaveValue('All categories');
  });
});
