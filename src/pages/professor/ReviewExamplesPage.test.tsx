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

let courseId = 'css-360-winter-2026-a7rp';
vi.mock('../../context/CourseContext', () => ({
  useCourseId: () => courseId,
}));

/**
 * Course metadata, per course.
 *
 * The durable starter-generation record lives here, so the mock is keyed by
 * course id: a page rendering one course must not see another's state.
 */
const metadataByCourse = new Map<string, CourseMetadata>();

vi.mock('../../hooks/useCourseMetadata', () => ({
  useCourseMetadata: (id: string | null) => {
    const metadata = (id && metadataByCourse.get(id)) || null;
    return {
      state: metadata
        ? ({ status: 'ready', metadata } as const)
        : ({ status: 'missing' } as const),
      metadata,
      retry: vi.fn(),
    };
  },
}));

import type { CourseMetadata, StoredStarterSeedGeneration } from '../../types';
import { ReviewExamplesPage } from './ReviewExamplesPage';

const BASE_METADATA: CourseMetadata = {
  name: 'CSS 350',
  title: 'Management Principles',
  term: 'Winter 2026',
  instructorName: '',
  createdAt: '2026-08-12T09:00:00.000Z',
  syllabusStatus: 'indexed',
  syllabusFileName: 'syllabus.pdf',
  syllabusType: 'pdf',
  chunkCount: 12,
};

function setGeneration(
  id: string,
  starterSeedGeneration: StoredStarterSeedGeneration | undefined,
) {
  metadataByCourse.set(id, { ...BASE_METADATA, ...(starterSeedGeneration ? { starterSeedGeneration } : {}) });
}

/** A course whose syllabus produced nothing yet. */
function noSeeds(id = courseId) {
  listCourseSeeds.mockResolvedValue({
    courseId: id,
    count: 0,
    firebasePath: `courses/${id}/seedExamples`,
    seeds: [],
  });
}

function renderPage() {
  return render(
    <MemoryRouter initialEntries={[`/course/${courseId}/review`]}>
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
    courseId = 'css-360-winter-2026-a7rp';
    metadataByCourse.clear();
    // Most courses have no generation record at all; the review flow must be
    // exactly what it was before this state existed.
    setGeneration(courseId, undefined);
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

  /*
   * Starter generation.
   *
   * The state is durable and lives on the course, so these tests set the
   * record rather than driving the page through a sequence — which is the
   * behaviour that matters: what a professor sees comes from what was stored,
   * not from what this tab happened to witness.
   */
  describe('starter example generation', () => {
    it('says examples are being generated instead of showing an empty course', async () => {
      noSeeds();
      setGeneration(courseId, { status: 'generating', targetCount: 50 });

      renderPage();

      expect(
        await screen.findByText('Generating starter examples…'),
      ).toBeInTheDocument();
      expect(
        screen.getByText(/creating example questions from your syllabus/i),
      ).toBeInTheDocument();
      expect(screen.getByText(/several minutes/i)).toBeInTheDocument();
      // The wording that made an upload look like it had done nothing.
      expect(
        screen.queryByText(/No example questions have been collected/i),
      ).not.toBeInTheDocument();
    });

    it('treats a queued job as generating', async () => {
      noSeeds();
      setGeneration(courseId, { status: 'queued', targetCount: 50 });

      renderPage();

      expect(
        await screen.findByText('Generating starter examples…'),
      ).toBeInTheDocument();
    });

    it('invents no progress figure', async () => {
      noSeeds();
      setGeneration(courseId, {
        status: 'generating',
        targetCount: 50,
        finalCount: 12,
        savedCount: 11,
      });

      renderPage();
      await screen.findByText('Generating starter examples…');

      expect(document.body.textContent ?? '').not.toMatch(
        /%|11 of 50|12 of 50|percent/i,
      );
    });

    it('keeps saying so while more examples are still arriving', async () => {
      setGeneration(courseId, { status: 'generating', targetCount: 50 });

      renderPage();

      // The queue is usable and the notice sits above it.
      expect(await screen.findByText('Can I submit late?')).toBeInTheDocument();
      expect(screen.getByText('Generating starter examples…')).toBeInTheDocument();
    });

    it('shows the normal review UI once generation is ready', async () => {
      setGeneration(courseId, {
        status: 'ready',
        targetCount: 50,
        savedCount: 48,
        completedAt: '2026-08-12T10:30:00.000Z',
      });

      renderPage();

      expect(await screen.findByText('Can I submit late?')).toBeInTheDocument();
      expect(countOnTab('Awaiting review')).toBe('1');
      expect(screen.queryByText('Generating starter examples…')).not.toBeInTheDocument();
    });

    it('shows the normal review UI for a partial run, which still has examples', async () => {
      setGeneration(courseId, { status: 'partial', targetCount: 50, savedCount: 9 });

      renderPage();

      expect(await screen.findByText('Can I submit late?')).toBeInTheDocument();
      expect(document.body.textContent ?? '').not.toMatch(/partial|couldn't create/i);
    });

    it('explains a failure in language a professor can act on', async () => {
      noSeeds();
      setGeneration(courseId, {
        status: 'failed',
        error: 'ollama request to 127.0.0.1:11434 timed out after 600s',
        completedAt: '2026-08-12T10:30:00.000Z',
      });

      renderPage();

      expect(
        await screen.findByText("We couldn't create starter examples"),
      ).toBeInTheDocument();
      expect(screen.getByText(/syllabus is saved/i)).toBeInTheDocument();
      expect(screen.getByText(/administrator/i)).toBeInTheDocument();
    });

    it('never shows a professor the recorded failure detail', async () => {
      noSeeds();
      setGeneration(courseId, {
        status: 'failed',
        error: 'ollama request to 127.0.0.1:11434 timed out after 600s',
      });

      renderPage();
      await screen.findByText("We couldn't create starter examples");

      const text = document.body.textContent ?? '';
      expect(text).not.toMatch(/ollama|11434|127\.0\.0\.1|timed out|600s/i);
      expect(text).not.toMatch(/traceback|exception|firebase|http/i);
    });

    it('offers no retry, because nothing here can safely start one', async () => {
      noSeeds();
      setGeneration(courseId, { status: 'failed' });

      renderPage();
      await screen.findByText("We couldn't create starter examples");

      for (const button of screen.getAllByRole('button')) {
        expect(button.textContent ?? '').not.toMatch(/try again|retry|regenerate/i);
      }
    });

    it('keeps one course’s generation state out of another’s page', async () => {
      const other = 'css-490-spring-2026-cgvl';
      setGeneration(other, { status: 'generating', targetCount: 50 });
      setGeneration(courseId, undefined);
      noSeeds();

      renderPage();

      expect(
        await screen.findByText(/No example questions have been collected/i),
      ).toBeInTheDocument();
      expect(screen.queryByText('Generating starter examples…')).not.toBeInTheDocument();
    });

    it('shows the same thing after a refresh, because the state is stored', async () => {
      noSeeds();
      setGeneration(courseId, { status: 'generating', targetCount: 50 });

      const first = renderPage();
      expect(
        await screen.findByText('Generating starter examples…'),
      ).toBeInTheDocument();

      // A refresh: nothing of this page survives, only the stored record.
      first.unmount();
      cleanup();
      renderPage();

      expect(
        await screen.findByText('Generating starter examples…'),
      ).toBeInTheDocument();
    });

    it('leaves a course with no generation record exactly as it was', async () => {
      noSeeds();
      setGeneration(courseId, undefined);

      renderPage();

      expect(
        await screen.findByText(/No example questions have been collected/i),
      ).toBeInTheDocument();
      expect(screen.queryByText('Generating starter examples…')).not.toBeInTheDocument();
      expect(
        screen.queryByText("We couldn't create starter examples"),
      ).not.toBeInTheDocument();
    });
  });
});
