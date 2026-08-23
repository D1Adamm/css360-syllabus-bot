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

let courseId = 'css-350-winter-2026-b3kq';
vi.mock('../../context/CourseContext', () => ({
  useCourseId: () => courseId,
}));

vi.mock('../../hooks/useCourseMetadata', () => ({
  useCourseMetadata: () => ({
    state: { status: 'missing' } as const,
    metadata: null,
    retry: vi.fn(),
  }),
}));

import { ReviewExamplesPage } from './ReviewExamplesPage';

interface SeedFixture {
  id: string;
  question: string;
  answer: string;
  category?: string;
  reviewStatus?: string;
  origin?: string;
  evidenceQuote?: string;
  sourceSection?: string;
  wasEdited?: boolean;
  originalQuestion?: string;
  originalAnswer?: string;
  normalizedQuestionKey?: string;
}

const SEEDS: SeedFixture[] = [
  {
    id: 'seed-1',
    question: 'Can I submit late?',
    answer: 'Late work may be submitted within 24 hours for a 10% penalty.',
    category: 'Late work',
    reviewStatus: 'generated',
    origin: 'ai_generated',
    evidenceQuote: 'Late work is accepted for 24 hours at a 10% penalty.',
    sourceSection: 'Policies',
  },
  {
    id: 'seed-2',
    question: 'Where are office hours?',
    answer: 'Office hours are Tuesdays at 2pm in UW1-360.',
    category: 'Contact',
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
  {
    id: 'seed-4',
    question: 'Where are office hours?',
    answer: 'Tuesdays at 2pm.',
    category: 'Contact',
    reviewStatus: 'approved',
    origin: 'ai_generated',
  },
  {
    id: 'seed-5',
    question: 'How do I contact the TA?',
    answer: 'Email the TA through Canvas.',
    category: 'Contact',
    reviewStatus: 'rejected',
    origin: 'user',
  },
  {
    id: 'seed-6',
    question: 'When is the final exam?',
    answer: 'The final exam is in week 10.',
    category: 'Schedule',
    reviewStatus: 'edited',
    wasEdited: true,
    originalQuestion: 'Final?',
    originalAnswer: 'Week 10.',
    origin: 'ai_generated',
  },
];

function loadSeeds(seeds: SeedFixture[] = SEEDS) {
  listCourseSeeds.mockResolvedValue({
    courseId,
    count: seeds.length,
    seeds,
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

function rows(): HTMLElement[] {
  return Array.from(document.querySelectorAll<HTMLElement>('.review-row'));
}

function rowFor(question: string): HTMLElement {
  const row = rows().find((element) =>
    element.querySelector('.review-row__question')?.textContent === question,
  );
  if (!row) {
    throw new Error(`No list row for "${question}"`);
  }
  return row;
}

function countOnTab(label: string): string {
  const tab = screen.getByRole('tab', { name: new RegExp(`^${label}`) });
  return tab.querySelector('.filter-tab__count')?.textContent ?? '';
}

function selectTab(label: string) {
  fireEvent.click(screen.getByRole('tab', { name: new RegExp(`^${label}`) }));
}

/** Echoes the request back, which is what the real endpoint does. */
function echoReview() {
  reviewCourseSeed.mockImplementation(
    async (
      _courseId: string,
      seedId: string,
      body: { reviewStatus: string; question?: string; answer?: string },
    ) => {
      const original = SEEDS.find((seed) => seed.id === seedId);
      const textChanged =
        body.question !== undefined || body.answer !== undefined;
      return {
        courseId,
        seedId,
        seed: {
          ...original,
          id: seedId,
          question: body.question ?? original?.question,
          answer: body.answer ?? original?.answer,
          reviewStatus: body.reviewStatus,
          status: body.reviewStatus,
          ...(textChanged
            ? {
                wasEdited: true,
                originalQuestion: original?.question,
                originalAnswer: original?.answer,
              }
            : {}),
        },
      };
    },
  );
}

describe('ReviewExamplesPage list view', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    courseId = 'css-350-winter-2026-b3kq';
    loadSeeds();
    echoReview();
  });

  afterEach(() => {
    cleanup();
  });

  describe('view modes', () => {
    it('opens in list view and renders every matching example at once', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      // All three awaiting-review examples, not one at a time.
      expect(rows()).toHaveLength(3);
      expect(screen.getByText('Where are office hours?')).toBeInTheDocument();
      expect(screen.getByText('What is the grading policy?')).toBeInTheDocument();
      // No one-at-a-time pager.
      expect(screen.queryByRole('button', { name: /Previous/ })).toBeNull();
    });

    it('numbers each row within the current filter', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      expect(within(rowFor('Can I submit late?')).getByText('1 of 3')).toBeInTheDocument();
      expect(
        within(rowFor('What is the grading policy?')).getByText('3 of 3'),
      ).toBeInTheDocument();
    });

    it('toggles to card view and back', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      fireEvent.click(screen.getByRole('button', { name: 'Card view' }));
      expect(rows()).toHaveLength(0);
      expect(document.querySelector('.review-card')).not.toBeNull();
      expect(screen.getByRole('button', { name: /Next/ })).toBeInTheDocument();

      fireEvent.click(screen.getByRole('button', { name: 'List view' }));
      expect(rows()).toHaveLength(3);
      expect(document.querySelector('.review-card')).toBeNull();
    });

    it('shows the status of every example, not just pending ones', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');
      selectTab('All');

      expect(rows()).toHaveLength(6);
      expect(
        within(rowFor('Can I submit late?')).getByText('Awaiting review'),
      ).toBeInTheDocument();
      // Two rows ask "Where are office hours?"; the approved one is the fourth.
      expect(within(rows()[3]!).getByText('Approved')).toBeInTheDocument();
      expect(
        within(rowFor('How do I contact the TA?')).getByText('Rejected'),
      ).toBeInTheDocument();
      expect(
        within(rowFor('When is the final exam?')).getByText('Edited'),
      ).toBeInTheDocument();
    });
  });

  describe('metadata on a row', () => {
    it('names the origin, category and source section', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      const row = rowFor('Can I submit late?');
      expect(
        within(row).getByText('AI generated · Late work · Policies'),
      ).toBeInTheDocument();
    });

    it('marks a student submission as such', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');
      selectTab('Rejected');

      expect(
        within(rowFor('How do I contact the TA?')).getByText(/Student submitted/),
      ).toBeInTheDocument();
    });

    it('flags questions that are word-for-word repeats', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');
      selectTab('All');

      // seed-2 and seed-4 ask exactly the same question.
      expect(screen.getAllByText('Possible duplicate')).toHaveLength(2);
    });
  });

  describe('expand and collapse', () => {
    it('collapses the supporting evidence, never the answer itself', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      const row = rowFor('Can I submit late?');
      // The answer is the thing being approved, so it reads in full collapsed.
      expect(
        within(row).getByText(
          'Late work may be submitted within 24 hours for a 10% penalty.',
        ),
      ).toBeInTheDocument();
      // The syllabus quote is supporting material and waits for an expand.
      expect(
        within(row).queryByText(
          'Late work is accepted for 24 hours at a 10% penalty.',
        ),
      ).toBeNull();
      // Actions are reachable while collapsed.
      expect(
        within(row).getByRole('button', { name: 'Approve example 1' }),
      ).toBeInTheDocument();

      fireEvent.click(within(row).getByRole('button', { name: /^Expand example 1/ }));

      expect(
        within(rowFor('Can I submit late?')).getByText(
          'Late work is accepted for 24 hours at a 10% penalty.',
        ),
      ).toBeInTheDocument();
    });

    it('renders a long answer whole, collapsed and expanded alike', async () => {
      const longAnswer = Array.from(
        { length: 12 },
        (_, i) =>
          `Sentence ${i + 1} of the late-work policy, spelled out at the kind of ` +
          'length a generated answer actually reaches in practice.',
      ).join(' ');

      loadSeeds([
        {
          id: 'seed-long',
          question: 'What exactly is the late work policy?',
          answer: longAnswer,
          category: 'Late work',
          reviewStatus: 'generated',
          origin: 'ai_generated',
        },
      ]);

      renderPage();
      await screen.findByText('What exactly is the late work policy?');

      const paragraph = document.querySelector('.review-row__answer');
      expect(paragraph).not.toBeNull();

      // Every character, collapsed. No slice, no ellipsis.
      expect(paragraph).toHaveTextContent(longAnswer);
      expect(paragraph?.textContent).toBe(longAnswer);
      expect(paragraph?.textContent).not.toMatch(/…|\.\.\./);

      // Nothing clips it either: no clamp, no fixed height, no hidden overflow.
      const styles = getComputedStyle(paragraph as Element);
      expect(styles.getPropertyValue('-webkit-line-clamp')).toBe('');
      expect(styles.getPropertyValue('line-clamp')).toBe('');
      expect(['', 'visible']).toContain(styles.overflow);
      expect(['', 'none', 'auto']).toContain(styles.maxHeight);
      expect(['', 'auto']).toContain(styles.height);

      fireEvent.click(screen.getByRole('button', { name: /^Expand example 1/ }));
      expect(
        document.querySelector('.review-row__answer')?.textContent,
      ).toBe(longAnswer);
    });

    it('expands and collapses one example without touching the others', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      fireEvent.click(
        within(rowFor('Can I submit late?')).getByRole('button', {
          name: /^Expand example 1/,
        }),
      );

      expect(document.querySelectorAll('.review-row--expanded')).toHaveLength(1);

      fireEvent.click(
        within(rowFor('Can I submit late?')).getByRole('button', {
          name: /^Collapse example 1/,
        }),
      );
      expect(document.querySelectorAll('.review-row--expanded')).toHaveLength(0);
    });

    it('expands and collapses everything at once', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      fireEvent.click(screen.getByRole('button', { name: 'Expand all' }));
      expect(document.querySelectorAll('.review-row--expanded')).toHaveLength(3);

      fireEvent.click(screen.getByRole('button', { name: 'Collapse all' }));
      expect(document.querySelectorAll('.review-row--expanded')).toHaveLength(0);
    });
  });

  describe('reviewing from the list', () => {
    it('approves a row through the existing review endpoint', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      fireEvent.click(
        within(rowFor('Can I submit late?')).getByRole('button', {
          name: 'Approve example 1',
        }),
      );

      await waitFor(() => {
        expect(reviewCourseSeed).toHaveBeenCalledWith(courseId, 'seed-1', {
          reviewStatus: 'approved',
        });
      });
    });

    it('rejects a row through the existing review endpoint', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      fireEvent.click(
        within(rowFor('Where are office hours?')).getByRole('button', {
          name: 'Reject example 2',
        }),
      );

      await waitFor(() => {
        expect(reviewCourseSeed).toHaveBeenCalledWith(courseId, 'seed-2', {
          reviewStatus: 'rejected',
        });
      });
    });

    it('updates the counts and leaves the rest of the list in place', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');
      expect(countOnTab('Awaiting review')).toBe('3');
      expect(countOnTab('Approved')).toBe('1');
      expect(screen.getByText('3 reviewed · 3 remaining')).toBeInTheDocument();

      fireEvent.click(
        within(rowFor('Can I submit late?')).getByRole('button', {
          name: 'Approve example 1',
        }),
      );

      await waitFor(() => {
        expect(countOnTab('Approved')).toBe('2');
      });
      expect(countOnTab('Awaiting review')).toBe('2');
      expect(screen.getByText('4 reviewed · 2 remaining')).toBeInTheDocument();

      // Still in list view, still scrolled through the same remaining rows.
      expect(rows()).toHaveLength(2);
      expect(screen.getByText('Where are office hours?')).toBeInTheDocument();
      expect(screen.getByText('What is the grading policy?')).toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /Next/ })).toBeNull();
    });

    it('never approves anything the professor did not click', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      expect(reviewCourseSeed).not.toHaveBeenCalled();
      expect(countOnTab('Awaiting review')).toBe('3');
    });
  });

  describe('inline editing', () => {
    async function openEditor(question: string, position: number) {
      fireEvent.click(
        within(rowFor(question)).getByRole('button', {
          name: `Edit example ${position}`,
        }),
      );
      return await screen.findByLabelText('Question');
    }

    it('edits the question and answer in place and saves', async () => {
      renderPage();
      await screen.findByText('What is the grading policy?');

      await openEditor('What is the grading policy?', 3);
      fireEvent.change(screen.getByLabelText('Question'), {
        target: { value: 'How does grading work?' },
      });
      fireEvent.change(screen.getByLabelText('Answer'), {
        target: { value: 'Grades come from two exams and one project.' },
      });
      fireEvent.click(screen.getByRole('button', { name: /^Save changes/ }));

      await waitFor(() => {
        expect(reviewCourseSeed).toHaveBeenCalledWith(courseId, 'seed-3', {
          reviewStatus: 'edited',
          question: 'How does grading work?',
          answer: 'Grades come from two exams and one project.',
          reviewNotes: undefined,
        });
      });

      // Still in list view; the edited example moved to Edited.
      await waitFor(() => {
        expect(countOnTab('Edited')).toBe('2');
      });
      expect(document.querySelector('.review-list')).not.toBeNull();
      expect(document.querySelector('.review-card')).toBeNull();

      selectTab('Edited');
      expect(screen.getByText('How does grading work?')).toBeInTheDocument();
    });

    it('discards the draft on cancel', async () => {
      renderPage();
      await screen.findByText('What is the grading policy?');

      await openEditor('What is the grading policy?', 3);
      fireEvent.change(screen.getByLabelText('Question'), {
        target: { value: 'Something else entirely' },
      });
      fireEvent.click(screen.getByRole('button', { name: /^Cancel editing/ }));

      expect(reviewCourseSeed).not.toHaveBeenCalled();
      expect(screen.getByText('What is the grading policy?')).toBeInTheDocument();
      expect(screen.queryByLabelText('Question')).toBeNull();

      // Reopening starts from what is stored, not from the discarded draft.
      await openEditor('What is the grading policy?', 3);
      expect(screen.getByLabelText('Question')).toHaveValue(
        'What is the grading policy?',
      );
    });

    it('keeps the editor open and explains a failed save', async () => {
      reviewCourseSeed.mockRejectedValue(new Error('boom'));
      renderPage();
      await screen.findByText('What is the grading policy?');

      await openEditor('What is the grading policy?', 3);
      fireEvent.change(screen.getByLabelText('Answer'), {
        target: { value: 'A better answer.' },
      });
      fireEvent.click(screen.getByRole('button', { name: /^Save changes/ }));

      expect(
        await screen.findByText(/could not be saved/i),
      ).toBeInTheDocument();
      // The professor keeps their words rather than losing them to a failure.
      expect(screen.getByLabelText('Answer')).toHaveValue('A better answer.');
      expect(countOnTab('Edited')).toBe('1');
    });
  });

  describe('bulk selection', () => {
    function selectExample(position: number) {
      fireEvent.click(screen.getByLabelText(`Select example ${position}`));
    }

    it('selects one and then several examples', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');
      expect(screen.queryByText(/selected$/)).toBeNull();

      selectExample(1);
      expect(screen.getByText('1 selected')).toBeInTheDocument();

      selectExample(2);
      expect(screen.getByText('2 selected')).toBeInTheDocument();
    });

    it('selects all visible examples and clears the selection', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      fireEvent.click(screen.getByRole('button', { name: 'Select all visible' }));
      expect(screen.getByText('3 selected')).toBeInTheDocument();

      fireEvent.click(screen.getByRole('button', { name: 'Clear selection' }));
      expect(screen.queryByText('3 selected')).toBeNull();
    });

    it('approves the selection with one request per example', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      fireEvent.click(screen.getByRole('button', { name: 'Select all visible' }));
      fireEvent.click(screen.getByRole('button', { name: 'Approve selected' }));

      await waitFor(() => {
        expect(reviewCourseSeed).toHaveBeenCalledTimes(3);
      });
      for (const seedId of ['seed-1', 'seed-2', 'seed-3']) {
        expect(reviewCourseSeed).toHaveBeenCalledWith(courseId, seedId, {
          reviewStatus: 'approved',
        });
      }

      await waitFor(() => {
        expect(countOnTab('Approved')).toBe('4');
      });
      expect(countOnTab('Awaiting review')).toBe('0');
      expect(screen.getByText('3 examples approved.')).toBeInTheDocument();
      // Everything succeeded, so nothing stays selected.
      expect(screen.queryByText(/\d+ selected/)).toBeNull();
    });

    it('confirms before rejecting a selection, then rejects it', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      fireEvent.click(screen.getByLabelText('Select example 1'));
      fireEvent.click(screen.getByLabelText('Select example 2'));
      fireEvent.click(screen.getByRole('button', { name: 'Reject selected' }));

      const dialog = await screen.findByRole('alertdialog');
      expect(within(dialog).getByText('Reject 2 examples?')).toBeInTheDocument();
      expect(reviewCourseSeed).not.toHaveBeenCalled();

      fireEvent.click(within(dialog).getByRole('button', { name: 'Reject selected' }));

      await waitFor(() => {
        expect(reviewCourseSeed).toHaveBeenCalledTimes(2);
      });
      expect(reviewCourseSeed).toHaveBeenCalledWith(courseId, 'seed-1', {
        reviewStatus: 'rejected',
      });
      await waitFor(() => {
        expect(countOnTab('Rejected')).toBe('3');
      });
    });

    it('cancels a bulk rejection without touching anything', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      fireEvent.click(screen.getByLabelText('Select example 1'));
      fireEvent.click(screen.getByRole('button', { name: 'Reject selected' }));
      const dialog = await screen.findByRole('alertdialog');
      fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));

      expect(reviewCourseSeed).not.toHaveBeenCalled();
      expect(screen.getByText('1 selected')).toBeInTheDocument();
    });

    it('reports partial failure honestly and keeps only the failures selected', async () => {
      reviewCourseSeed.mockImplementation(
        async (_courseId: string, seedId: string, body: { reviewStatus: string }) => {
          if (seedId === 'seed-2') {
            throw new Error('boom');
          }
          const original = SEEDS.find((seed) => seed.id === seedId);
          return {
            courseId,
            seedId,
            seed: {
              ...original,
              id: seedId,
              reviewStatus: body.reviewStatus,
              status: body.reviewStatus,
            },
          };
        },
      );

      renderPage();
      await screen.findByText('Can I submit late?');

      fireEvent.click(screen.getByRole('button', { name: 'Select all visible' }));
      fireEvent.click(screen.getByRole('button', { name: 'Approve selected' }));

      expect(
        await screen.findByText('2 approved. 1 could not be saved and is unchanged.'),
      ).toBeInTheDocument();

      // The two that worked are approved; the one that failed is untouched.
      expect(countOnTab('Approved')).toBe('3');
      expect(countOnTab('Awaiting review')).toBe('1');
      expect(screen.getByText('Where are office hours?')).toBeInTheDocument();
      expect(
        within(rowFor('Where are office hours?')).getByText('Awaiting review'),
      ).toBeInTheDocument();

      // The failure stays selected so it can be retried.
      expect(screen.getByText('1 selected')).toBeInTheDocument();
    });

    it('does not fire one request per example all at once', async () => {
      let inFlight = 0;
      let peak = 0;
      reviewCourseSeed.mockImplementation(
        async (_courseId: string, seedId: string, body: { reviewStatus: string }) => {
          inFlight += 1;
          peak = Math.max(peak, inFlight);
          await Promise.resolve();
          inFlight -= 1;
          return {
            courseId,
            seedId,
            seed: { id: seedId, reviewStatus: body.reviewStatus, status: body.reviewStatus },
          };
        },
      );

      const many = Array.from({ length: 20 }, (_, i) => ({
        id: `bulk-${i}`,
        question: `Question ${i}?`,
        answer: `Answer ${i}.`,
        reviewStatus: 'generated',
        origin: 'ai_generated',
      }));
      loadSeeds(many);

      renderPage();
      await screen.findByText('Question 0?');

      fireEvent.click(screen.getByRole('button', { name: 'Select all visible' }));
      fireEvent.click(screen.getByRole('button', { name: 'Approve selected' }));

      await waitFor(() => {
        expect(reviewCourseSeed).toHaveBeenCalledTimes(20);
      });
      expect(peak).toBeLessThanOrEqual(4);
    });
  });

  describe('filtering and search', () => {
    it('shows each review state under its own tab', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      expect(rows()).toHaveLength(3);

      selectTab('Approved');
      expect(rows()).toHaveLength(1);
      expect(screen.getByText('Where are office hours?')).toBeInTheDocument();

      selectTab('Edited');
      expect(rows()).toHaveLength(1);
      expect(screen.getByText('When is the final exam?')).toBeInTheDocument();

      selectTab('Rejected');
      expect(rows()).toHaveLength(1);
      expect(screen.getByText('How do I contact the TA?')).toBeInTheDocument();

      selectTab('All');
      expect(rows()).toHaveLength(6);
    });

    it('searches the question text', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');
      selectTab('All');

      fireEvent.change(screen.getByLabelText('Search examples'), {
        target: { value: 'office hours' },
      });

      expect(rows()).toHaveLength(2);
      expect(screen.queryByText('Can I submit late?')).toBeNull();
    });

    it('searches the answer text', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');
      selectTab('All');

      fireEvent.change(screen.getByLabelText('Search examples'), {
        target: { value: 'UW1-360' },
      });

      expect(rows()).toHaveLength(1);
      expect(screen.getByText('Where are office hours?')).toBeInTheDocument();
    });

    it('explains an empty search instead of claiming there is nothing to review', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      fireEvent.change(screen.getByLabelText('Search examples'), {
        target: { value: 'zzzz' },
      });

      expect(screen.getByText('No matching examples')).toBeInTheDocument();
      expect(screen.queryByText('Nothing waiting for you')).toBeNull();
    });

    it('filters by category', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');
      selectTab('All');

      fireEvent.change(screen.getByLabelText('Category'), {
        target: { value: 'Contact' },
      });

      expect(rows()).toHaveLength(3);
      expect(screen.queryByText('Can I submit late?')).toBeNull();
    });

    it('hides examples that have already been reviewed', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');
      selectTab('All');
      expect(rows()).toHaveLength(6);

      fireEvent.click(screen.getByLabelText('Hide reviewed'));

      expect(rows()).toHaveLength(3);
      expect(screen.queryByText('When is the final exam?')).toBeNull();
      expect(screen.queryByText('How do I contact the TA?')).toBeNull();
    });
  });

  describe('keyboard workflow', () => {
    it('moves the highlight with the arrow keys and approves the highlighted row', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      fireEvent.keyDown(document, { key: 'ArrowDown' });
      expect(rowFor('Can I submit late?')).toHaveClass('review-row--focused');

      fireEvent.keyDown(document, { key: 'ArrowDown' });
      expect(rowFor('Where are office hours?')).toHaveClass('review-row--focused');

      fireEvent.keyDown(document, { key: 'a' });
      await waitFor(() => {
        expect(reviewCourseSeed).toHaveBeenCalledWith(courseId, 'seed-2', {
          reviewStatus: 'approved',
        });
      });
    });

    it('rejects the highlighted row with r and opens the editor with e', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      fireEvent.keyDown(document, { key: 'j' });
      fireEvent.keyDown(document, { key: 'e' });
      expect(screen.getByLabelText('Question')).toHaveValue('Can I submit late?');

      fireEvent.click(screen.getByRole('button', { name: /^Cancel editing/ }));
      fireEvent.keyDown(document, { key: 'r' });
      await waitFor(() => {
        expect(reviewCourseSeed).toHaveBeenCalledWith(courseId, 'seed-1', {
          reviewStatus: 'rejected',
        });
      });
    });

    it('does not fire shortcuts while typing in the inline editor', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      fireEvent.click(
        within(rowFor('Can I submit late?')).getByRole('button', {
          name: 'Edit example 1',
        }),
      );
      const question = await screen.findByLabelText('Question');

      fireEvent.keyDown(question, { key: 'a' });
      fireEvent.keyDown(question, { key: 'r' });
      fireEvent.keyDown(question, { key: 'e' });
      fireEvent.keyDown(question, { key: 'j' });

      expect(reviewCourseSeed).not.toHaveBeenCalled();
      expect(screen.getByLabelText('Question')).toBeInTheDocument();
    });

    it('does not fire shortcuts while typing in the search box', async () => {
      renderPage();
      await screen.findByText('Can I submit late?');

      const search = screen.getByLabelText('Search examples');
      fireEvent.keyDown(search, { key: 'a' });
      fireEvent.keyDown(search, { key: 'r' });

      expect(reviewCourseSeed).not.toHaveBeenCalled();
    });
  });

  describe('course scoping', () => {
    it('reviews against the course in context, never another one', async () => {
      courseId = 'css-360-winter-2026-a7rp';
      loadSeeds();
      renderPage();
      await screen.findByText('Can I submit late?');

      expect(listCourseSeeds).toHaveBeenCalledWith('css-360-winter-2026-a7rp');

      fireEvent.click(screen.getByRole('button', { name: 'Select all visible' }));
      fireEvent.click(screen.getByRole('button', { name: 'Approve selected' }));

      await waitFor(() => {
        expect(reviewCourseSeed).toHaveBeenCalledTimes(3);
      });
      for (const call of reviewCourseSeed.mock.calls) {
        expect(call[0]).toBe('css-360-winter-2026-a7rp');
      }
    });
  });
});
