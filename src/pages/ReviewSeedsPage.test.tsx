/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

const listCourseSeeds = vi.fn();
const reviewCourseSeed = vi.fn();
const exportApprovedCourseSeeds = vi.fn();
const getApprovedExportStatus = vi.fn();
const prepareTrainingSplit = vi.fn();

vi.mock('../lib/api', () => ({
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
  exportApprovedCourseSeeds: (...args: unknown[]) =>
    exportApprovedCourseSeeds(...args),
  getApprovedExportStatus: (...args: unknown[]) =>
    getApprovedExportStatus(...args),
  prepareTrainingSplit: (...args: unknown[]) => prepareTrainingSplit(...args),
}));

vi.mock('../context/CourseContext', () => ({
  useCourseId: () => 'css-360-winter-2026-a7rp',
}));

import { ReviewSeedsPage } from './ReviewSeedsPage';

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/course/css-360-winter-2026-a7rp/review']}>
      <Routes>
        <Route path="/course/:courseId/review" element={<ReviewSeedsPage />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('ReviewSeedsPage', () => {
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
    exportApprovedCourseSeeds.mockResolvedValue({
      courseId: 'css-360-winter-2026-a7rp',
      summary: {
        approvedCount: 1,
        exportedCount: 1,
        validatedCount: 1,
        validationPassed: true,
        exportPath: 'data/exports/css-360-winter-2026-a7rp/approved-finetune.jsonl',
        files: {
          finetuneJsonl: 'data/exports/css-360-winter-2026-a7rp/approved-finetune.jsonl',
        },
      },
    });
    getApprovedExportStatus.mockResolvedValue({
      courseId: 'css-360-winter-2026-a7rp',
      exists: false,
      exportPath: 'data/exports/css-360-winter-2026-a7rp/approved-finetune.jsonl',
      exampleCount: 0,
      sourceFile: 'approved-finetune.jsonl',
    });
    prepareTrainingSplit.mockResolvedValue({
      courseId: 'css-360-winter-2026-a7rp',
      summary: {
        trainExamples: 48,
        validationExamples: 6,
        totalExamples: 54,
        splitSeed: 360,
      },
    });
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
    expect(screen.getByText(/Generated:/)).toHaveTextContent('Generated: 1');
    expect(screen.getByText(/Approved:/)).toHaveTextContent('Approved: 1');
    expect(
      screen.getByText(/courses\/css-360-winter-2026-a7rp\/seedExamples/),
    ).toBeInTheDocument();
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
    fireEvent.click(screen.getByRole('tab', { name: 'Generated' }));
    expect(screen.getByText('Can I submit late?')).toBeInTheDocument();
    expect(screen.getByText(/Generated:/)).toHaveTextContent('Generated: 1');
    expect(screen.getByText(/Approved:/)).toHaveTextContent('Approved: 1');

    fireEvent.click(screen.getByRole('button', { name: 'Approve' }));

    await waitFor(() => {
      expect(screen.queryByText('Can I submit late?')).not.toBeInTheDocument();
    });
    expect(screen.getByText(/Generated:/)).toHaveTextContent('Generated: 0');
    expect(screen.getByText(/Approved:/)).toHaveTextContent('Approved: 2');
    expect(screen.getByText(/No seeds match this review filter/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('tab', { name: 'Approved' }));
    expect(screen.getByText('Can I submit late?')).toBeInTheDocument();
    expect(screen.getByText('Where are office hours?')).toBeInTheDocument();
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
    fireEvent.click(screen.getByRole('tab', { name: 'Generated' }));
    expect(screen.getByText(/Generated:/)).toHaveTextContent('Generated: 2');

    const rejectButtons = screen.getAllByRole('button', { name: 'Reject' });
    fireEvent.click(rejectButtons[0]);
    await waitFor(() => {
      expect(screen.queryByText('Can I submit late?')).not.toBeInTheDocument();
    });
    expect(screen.getByText(/Generated:/)).toHaveTextContent('Generated: 1');
    expect(screen.getByText(/Rejected:/)).toHaveTextContent('Rejected: 1');
    expect(screen.getByText('What is the grading policy?')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Edit' }));
    fireEvent.change(screen.getByLabelText('Question'), {
      target: { value: 'How does grading work?' },
    });
    fireEvent.change(screen.getByLabelText('Answer'), {
      target: { value: 'Grades are based on exams and projects with clear weights.' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Save edit' }));

    await waitFor(() => {
      expect(screen.queryByText('What is the grading policy?')).not.toBeInTheDocument();
      expect(screen.queryByText('How does grading work?')).not.toBeInTheDocument();
    });
    expect(screen.getByText(/Generated:/)).toHaveTextContent('Generated: 0');
    expect(screen.getByText(/Edited:/)).toHaveTextContent('Edited: 1');

    fireEvent.click(screen.getByRole('tab', { name: 'Edited' }));
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
    await screen.findByText('How does grading work?');
    fireEvent.click(screen.getByRole('tab', { name: 'Edited' }));
    expect(screen.getByText('How does grading work?')).toBeInTheDocument();
    expect(screen.getByText(/Edited:/)).toHaveTextContent('Edited: 1');
    expect(screen.getByText(/Approved:/)).toHaveTextContent('Approved: 0');

    fireEvent.click(screen.getByRole('button', { name: 'Approve' }));

    await waitFor(() => {
      expect(screen.queryByText('How does grading work?')).not.toBeInTheDocument();
    });
    expect(screen.getByText(/Edited:/)).toHaveTextContent('Edited: 0');
    expect(screen.getByText(/Approved:/)).toHaveTextContent('Approved: 1');

    fireEvent.click(screen.getByRole('tab', { name: 'Approved' }));
    expect(screen.getByText('How does grading work?')).toBeInTheDocument();
    expect(screen.getByText('approved')).toBeInTheDocument();
    expect(
      document.querySelector('.review-seed-card__status--edited'),
    ).toHaveTextContent('Edited');
    expect(screen.getByText('What is the grading policy?')).toBeInTheDocument();
    expect(
      screen.getByText('Grades are based on exams and projects.'),
    ).toBeInTheDocument();
  });

  it('filters by approved status', async () => {
    renderPage();
    await screen.findByText('Can I submit late?');
    expect(screen.getByLabelText('Seed number 1')).toHaveTextContent('Seed #1');
    expect(screen.getByLabelText('Seed number 2')).toHaveTextContent('Seed #2');
    fireEvent.click(screen.getByRole('tab', { name: 'Approved' }));
    expect(screen.queryByText('Can I submit late?')).not.toBeInTheDocument();
    expect(screen.getByText('Where are office hours?')).toBeInTheDocument();
    expect(screen.getByLabelText('Seed number 1')).toHaveTextContent('Seed #1');
    expect(screen.queryByLabelText('Seed number 2')).not.toBeInTheDocument();
  });

  it('exports approved seeds', async () => {
    renderPage();
    await screen.findByText('Can I submit late?');
    fireEvent.click(screen.getByRole('button', { name: 'Export Approved' }));
    await waitFor(() => {
      expect(exportApprovedCourseSeeds).toHaveBeenCalledWith(
        'css-360-winter-2026-a7rp',
      );
    });
    expect(
      await screen.findByText(
        /Exported and validated 1 approved seed → data\/exports\/css-360-winter-2026-a7rp\/approved-finetune\.jsonl/,
      ),
    ).toBeInTheDocument();
  });

  it('disables Prepare Training Split until an approved export exists', async () => {
    renderPage();
    await screen.findByText('Can I submit late?');
    await waitFor(() => {
      expect(getApprovedExportStatus).toHaveBeenCalledWith(
        'css-360-winter-2026-a7rp',
      );
    });
    expect(
      screen.getByRole('button', { name: 'Prepare Training Split' }),
    ).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: 'Export Approved' }));
    await screen.findByText(/Exported and validated 1 approved seed/);
    expect(
      screen.getByRole('button', { name: 'Prepare Training Split' }),
    ).not.toBeDisabled();
  });

  it('prepares a training split after an approved export exists', async () => {
    getApprovedExportStatus.mockResolvedValue({
      courseId: 'css-360-winter-2026-a7rp',
      exists: true,
      exportPath: 'data/exports/css-360-winter-2026-a7rp/approved-finetune.jsonl',
      exampleCount: 54,
      sourceFile: 'approved-finetune.jsonl',
    });

    renderPage();
    await screen.findByText('Can I submit late?');
    await waitFor(() => {
      expect(
        screen.getByRole('button', { name: 'Prepare Training Split' }),
      ).not.toBeDisabled();
    });

    fireEvent.click(screen.getByRole('button', { name: 'Prepare Training Split' }));
    await waitFor(() => {
      expect(prepareTrainingSplit).toHaveBeenCalledWith(
        'css-360-winter-2026-a7rp',
      );
    });
    expect(
      await screen.findByText('Prepared training split: 48 train, 6 validation'),
    ).toBeInTheDocument();
  });
});
