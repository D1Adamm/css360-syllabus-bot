/** @vitest-environment jsdom */
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

const listCourseSeeds = vi.fn();
const reviewCourseSeed = vi.fn();

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
  reviewCourseSeed: (...args: unknown[]) => reviewCourseSeed(...args),
}));

vi.mock('../../context/CourseContext', () => ({
  useCourseId: () => 'css-360-winter-2026-a7rp',
}));

import { ReviewExamplesPage } from './ReviewExamplesPage';

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/course/css-360-winter-2026-a7rp/review']}>
      <Routes>
        <Route path="/course/:courseId/review" element={<ReviewExamplesPage />} />
      </Routes>
    </MemoryRouter>,
  );
}

/** The count badge rendered inside a filter tab. */
function countOnTab(label: string): string {
  const tab = screen.getByRole('tab', { name: new RegExp(`^${label}`) });
  return tab.querySelector('.filter-tab__count')?.textContent ?? '';
}

describe('ReviewExamplesPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    listCourseSeeds.mockResolvedValue({
      courseId: 'css-360-winter-2026-a7rp',
      count: 2,
      firebasePath: 'courses/css-360-winter-2026-a7rp/seedExamples',
      seeds: [
        {
          id: 'seed-1',
          question: 'Can I submit late?',
          answer: 'Late work may be submitted within 24 hours.',
          category: 'Late work',
          evidenceQuote: 'Late work may be submitted within 24 hours.',
          factId: 'fact-01',
          reviewStatus: 'generated',
          origin: 'ai_generated',
          validation: { score: 0.9, reason: 'Grounded.' },
        },
        {
          id: 'seed-2',
          question: 'Where are office hours?',
          answer: 'Office hours are Tuesdays at 2pm.',
          category: 'Contact',
          reviewStatus: 'approved',
          origin: 'ai_generated',
          validation: { score: 0.88, reason: 'Useful.' },
        },
      ],
    });
    reviewCourseSeed.mockImplementation(
      async (
        _courseId: string,
        seedId: string,
        body: {
          reviewStatus: string;
          question?: string;
          answer?: string;
          reviewNotes?: string;
        },
      ) => {
        const textChanged =
          body.question !== undefined || body.answer !== undefined;
        const wasEdited =
          body.reviewStatus === 'edited' || textChanged;
        return {
          courseId: 'css-360-winter-2026-a7rp',
          seedId,
          firebasePath: `courses/css-360-winter-2026-a7rp/seedExamples/${seedId}`,
          seed: {
            id: seedId,
            question: body.question ?? 'Can I submit late?',
            answer: body.answer ?? 'Late work may be submitted within 24 hours.',
            instruction: body.question ?? 'Can I submit late?',
            response: body.answer ?? 'Late work may be submitted within 24 hours.',
            category: 'Late work',
            reviewStatus: body.reviewStatus,
            status: body.reviewStatus,
            reviewNotes: body.reviewNotes,
            origin: 'ai_generated',
            validation: { score: 0.9, reason: 'Grounded.' },
            ...(wasEdited
              ? {
                  wasEdited: true,
                  originalQuestion: 'What is the grading policy?',
                  originalAnswer: 'Grades are based on exams and projects.',
                }
              : {}),
          },
        };
      },
    );
  });

  afterEach(() => {
    cleanup();
  });

  it('loads course-scoped seeds and shows counts', async () => {
    renderPage();
    await waitFor(() => {
      expect(listCourseSeeds).toHaveBeenCalledWith('css-360-winter-2026-a7rp');
    });
    expect(await screen.findByText('Can I submit late?')).toBeInTheDocument();
    expect(countOnTab('Awaiting review')).toBe('1');
    expect(countOnTab('Approved')).toBe('1');
    // The storage path is no longer shown to professors.
    expect(
      screen.queryByText(/courses\/css-360-winter-2026-a7rp\/seedExamples/),
    ).not.toBeInTheDocument();
  });

  it('approves a seed via the review endpoint', async () => {
    renderPage();
    await screen.findByText('Can I submit late?');
    const approveButtons = screen.getAllByRole('button', { name: 'Approve' });
    fireEvent.click(approveButtons[0]);
    await waitFor(() => {
      expect(reviewCourseSeed).toHaveBeenCalledWith(
        'css-360-winter-2026-a7rp',
        'seed-1',
        { reviewStatus: 'approved' },
      );
    });
  });

  it('removes approved seed from Generated filter and updates counts immediately', async () => {
    renderPage();
    await screen.findByText('Can I submit late?');
    fireEvent.click(screen.getByRole('tab', { name: /^Awaiting review/ }));
    expect(screen.getByText('Can I submit late?')).toBeInTheDocument();
    expect(countOnTab('Awaiting review')).toBe('1');
    expect(countOnTab('Approved')).toBe('1');

    fireEvent.click(screen.getByRole('button', { name: 'Approve' }));

    await waitFor(() => {
      expect(screen.queryByText('Can I submit late?')).not.toBeInTheDocument();
    });
    expect(countOnTab('Awaiting review')).toBe('0');
    expect(countOnTab('Approved')).toBe('2');
    expect(screen.getByText('Nothing waiting for you')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('tab', { name: /^Approved/ }));
    // The queue shows one example at a time; both are now in the Approved list.
    expect(screen.getByText('1 of 2')).toBeInTheDocument();
    expect(screen.getByText('Can I submit late?')).toBeInTheDocument();
  });

  it('moves rejected and edited seeds out of Generated filter', async () => {
    listCourseSeeds.mockResolvedValue({
      courseId: 'css-360-winter-2026-a7rp',
      count: 2,
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
        {
          id: 'seed-3',
          question: 'What is the grading policy?',
          answer: 'Grades are based on exams and projects.',
          category: 'Grading',
          reviewStatus: 'generated',
          origin: 'ai_generated',
        },
      ],
    });

    renderPage();
    await screen.findByText('Can I submit late?');
    fireEvent.click(screen.getByRole('tab', { name: /^Awaiting review/ }));
    expect(countOnTab('Awaiting review')).toBe('2');

    const rejectButtons = screen.getAllByRole('button', { name: 'Reject' });
    fireEvent.click(rejectButtons[0]);
    await waitFor(() => {
      expect(screen.queryByText('Can I submit late?')).not.toBeInTheDocument();
    });
    expect(countOnTab('Awaiting review')).toBe('1');
    expect(countOnTab('Rejected')).toBe('1');
    expect(screen.getByText('What is the grading policy?')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Edit' }));
    fireEvent.change(screen.getByLabelText('Question'), {
      target: { value: 'How does grading work?' },
    });
    fireEvent.change(screen.getByLabelText('Answer'), {
      target: { value: 'Grades are based on exams and projects with clear weights.' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));

    await waitFor(() => {
      expect(screen.queryByText('What is the grading policy?')).not.toBeInTheDocument();
      expect(screen.queryByText('How does grading work?')).not.toBeInTheDocument();
    });
    expect(countOnTab('Awaiting review')).toBe('0');
    expect(countOnTab('Edited')).toBe('1');

    fireEvent.click(screen.getByRole('tab', { name: /^Edited/ }));
    expect(screen.getByText('How does grading work?')).toBeInTheDocument();
  });

  it('approving an edited seed moves it to Approved and keeps Edited badge', async () => {
    listCourseSeeds.mockResolvedValue({
      courseId: 'css-360-winter-2026-a7rp',
      count: 1,
      firebasePath: 'courses/css-360-winter-2026-a7rp/seedExamples',
      seeds: [
        {
          id: 'seed-edit',
          question: 'How does grading work?',
          answer: 'Grades are based on exams and projects with clear weights.',
          category: 'Grading',
          reviewStatus: 'edited',
          wasEdited: true,
          originalQuestion: 'What is the grading policy?',
          originalAnswer: 'Grades are based on exams and projects.',
          origin: 'ai_generated',
        },
      ],
    });
    reviewCourseSeed.mockResolvedValue({
      courseId: 'css-360-winter-2026-a7rp',
      seedId: 'seed-edit',
      firebasePath: 'courses/css-360-winter-2026-a7rp/seedExamples/seed-edit',
      seed: {
        id: 'seed-edit',
        question: 'How does grading work?',
        answer: 'Grades are based on exams and projects with clear weights.',
        category: 'Grading',
        reviewStatus: 'approved',
        status: 'approved',
        wasEdited: true,
        originalQuestion: 'What is the grading policy?',
        originalAnswer: 'Grades are based on exams and projects.',
        origin: 'ai_generated',
      },
    });

    renderPage();
    await waitFor(() => {
      expect(countOnTab('Edited')).toBe('1');
    });
    fireEvent.click(screen.getByRole('tab', { name: /^Edited/ }));
    expect(screen.getByText('How does grading work?')).toBeInTheDocument();
    expect(countOnTab('Edited')).toBe('1');
    expect(countOnTab('Approved')).toBe('0');

    fireEvent.click(screen.getByRole('button', { name: 'Approve' }));

    await waitFor(() => {
      expect(screen.queryByText('How does grading work?')).not.toBeInTheDocument();
    });
    expect(countOnTab('Edited')).toBe('0');
    expect(countOnTab('Approved')).toBe('1');

    fireEvent.click(screen.getByRole('tab', { name: /^Approved/ }));
    expect(screen.getByText('How does grading work?')).toBeInTheDocument();
    // Status and provenance both show on the card ("Approved" also names a tab).
    const card = document.querySelector('.review-card') as HTMLElement;
    expect(within(card).getByText('Approved')).toBeInTheDocument();
    // The edit provenance survives approval and is still reachable.
    expect(within(card).getByText('Edited')).toBeInTheDocument();
    fireEvent.click(
      screen.getByText('What this said before you edited it'),
    );
    expect(screen.getByText('What is the grading policy?')).toBeInTheDocument();
    expect(
      screen.getByText('Grades are based on exams and projects.'),
    ).toBeInTheDocument();
  });

  it('filters by approved status', async () => {
    renderPage();
    await screen.findByText('Can I submit late?');
    expect(screen.getByText('1 of 1')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('tab', { name: /^Approved/ }));
    expect(screen.queryByText('Can I submit late?')).not.toBeInTheDocument();
    expect(screen.getByText('Where are office hours?')).toBeInTheDocument();
    expect(screen.getByText('1 of 1')).toBeInTheDocument();
  });

  it('moves through the queue with the previous and next controls', async () => {
    listCourseSeeds.mockResolvedValue({
      courseId: 'css-360-winter-2026-a7rp',
      count: 2,
      firebasePath: 'courses/css-360-winter-2026-a7rp/seedExamples',
      seeds: [
        {
          id: 'seed-1',
          question: 'Can I submit late?',
          answer: 'Late work may be submitted within 24 hours.',
          reviewStatus: 'generated',
          origin: 'ai_generated',
        },
        {
          id: 'seed-3',
          question: 'What is the grading policy?',
          answer: 'Grades are based on exams and projects.',
          reviewStatus: 'generated',
          origin: 'ai_generated',
        },
      ],
    });

    renderPage();
    await screen.findByText('Can I submit late?');
    expect(screen.getByText('1 of 2')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Previous/ })).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: /Next/ }));
    expect(screen.getByText('2 of 2')).toBeInTheDocument();
    expect(screen.getByText('What is the grading policy?')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Next/ })).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: /Previous/ }));
    expect(screen.getByText('Can I submit late?')).toBeInTheDocument();
  });

});
