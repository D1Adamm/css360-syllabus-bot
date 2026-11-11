/** @vitest-environment jsdom */
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

/**
 * The shared Classroom answers page, as a professor and as an administrator.
 *
 * The backend decides what is prepared and what may be started; these tests
 * hold what the page does with that: real counts, exactly one action that fits
 * the state, a selection that is saved as it is made, and a status that comes
 * from the server every time (so a refresh or a return shows the same thing).
 */

let courseId = 'css-360-winter-2026-a7rp';
vi.mock('../../context/CourseContext', () => ({
  useCourseId: () => courseId,
}));

const api = vi.hoisted(() => ({
  fetchPreparationStatus: vi.fn(),
  savePreparationSelection: vi.fn(),
  startPreparation: vi.fn(),
  stopPreparation: vi.fn(),
}));

vi.mock('../../lib/classroomPreparation', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../lib/classroomPreparation')>()),
  ...api,
}));

import { RUNNING_POLL_MS } from '../../hooks/useClassroomPreparation';
import { ApiError } from '../../lib/httpClient';
import type { PreparationStatus } from '../../lib/classroomPreparation';
import { eightyOfOneTwenty, preparation, question } from '../../test/preparationFixtures';
import { ClassroomPreparationPage } from './ClassroomPreparationPage';

function renderPage(audience: 'professor' | 'admin' = 'professor') {
  return render(
    <MemoryRouter>
      <ClassroomPreparationPage audience={audience} />
    </MemoryRouter>,
  );
}

async function shown(status: PreparationStatus, audience: 'professor' | 'admin' = 'professor') {
  api.fetchPreparationStatus.mockResolvedValue(status);
  renderPage(audience);
  await screen.findByRole('heading', { name: 'Preparation' });
}

const RUNNING = eightyOfOneTwenty({
  state: 'preparing',
  running: true,
  canStart: false,
  startBlockedReason: 'running',
  current: { seedId: 's1', condition: 'fineTunedRag' },
  message: 'Preparing classroom answers. Currently generating Fine-Tuned + RAG.',
});

/** Every button in the summary: the page must never offer two ways forward. */
function summaryButtons(): string[] {
  const summary = screen.getByRole('heading', { name: 'Preparation' }).closest('section') as HTMLElement;
  return within(summary)
    .queryAllByRole('button')
    .map((button) => button.textContent ?? '');
}

describe('ClassroomPreparationPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    courseId = 'css-360-winter-2026-a7rp';
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  describe('summary', () => {
    it('shows the real counts, including questions fully prepared', async () => {
      await shown(eightyOfOneTwenty());

      expect(screen.getByText('Partially ready')).toBeInTheDocument();
      expect(
        screen.getByText('20 of 30 questions fully ready. Students can already use every prepared answer.'),
      ).toBeInTheDocument();
      const stat = (label: string) => screen.getByText(label).closest('.prep-stat') as HTMLElement;
      expect(stat('Answers ready')).toHaveTextContent('80 of 120');
      expect(stat('Questions fully prepared')).toHaveTextContent('20 of 30');
      expect(stat('Still to prepare')).toHaveTextContent('40');
      const bar = screen.getByRole('progressbar', { name: 'Answers ready' });
      expect(bar).toHaveAttribute('aria-valuenow', '80');
      expect(bar).toHaveAttribute('aria-valuemax', '120');
      expect(bar).toHaveAttribute('aria-valuetext', '80 of 120 answers ready');
    });

    it('does not derive "fully prepared" from the answer count', async () => {
      // 80 answers ready, but spread so that only 11 questions have all four.
      await shown(
        eightyOfOneTwenty({ counts: { ...eightyOfOneTwenty().counts, fullyPreparedQuestions: 11 } }),
      );
      const stat = screen.getByText('Questions fully prepared').closest('.prep-stat') as HTMLElement;
      expect(stat).toHaveTextContent('11 of 30');
    });

    it('says how many answers are out of date', async () => {
      await shown(
        eightyOfOneTwenty({
          counts: { ...eightyOfOneTwenty().counts, outdatedAnswers: 12 },
          notices: ['12 answers out of date because the syllabus or a model changed. Prepare the remaining answers to refresh them.'],
        }),
      );
      expect(screen.getByText('(12 out of date)')).toBeInTheDocument();
      expect(screen.getByText(/12 answers out of date because the syllabus or a model changed/)).toBeInTheDocument();
    });

    it('shows no counts or progress bar before anything is selected', async () => {
      await shown(
        preparation({
          canStart: false,
          startBlockedReason: 'nothing_selected',
          counts: { ...preparation().counts, selectedQuestions: 0, expectedAnswers: 0, missingAnswers: 0 },
          message: 'Choose the questions you want ready for class, then prepare their answers.',
          questions: preparation().questions.map((q) => ({ ...q, selected: false, answers: null })),
        }),
      );
      expect(screen.getByText('Not started')).toBeInTheDocument();
      expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
      expect(screen.queryByText('Answers ready')).not.toBeInTheDocument();
      expect(summaryButtons()).toEqual([]);
    });

    it('explains in plain words why something cannot be prepared', async () => {
      await shown(
        preparation({
          notices: [
            'This course has no active fine-tuned model, so only Base and RAG answers are prepared.',
            'Classroom answers are switched off on this server, so prepared answers are not used yet.',
          ],
        }),
      );
      expect(screen.getByText(/no active fine-tuned model/)).toBeInTheDocument();
      expect(screen.getByText(/switched off on this server/)).toBeInTheDocument();
    });
  });

  describe('the one action', () => {
    it('offers Prepare answers when nothing is prepared, and starts on click', async () => {
      await shown(preparation());
      expect(summaryButtons()).toEqual(['Prepare answers']);

      api.startPreparation.mockResolvedValue(RUNNING);
      fireEvent.click(screen.getByRole('button', { name: 'Prepare answers' }));

      expect(api.startPreparation).toHaveBeenCalledWith(courseId);
      expect(await screen.findByRole('button', { name: 'Stop preparation' })).toBeInTheDocument();
      expect(screen.getByText('Preparing')).toBeInTheDocument();
      expect(screen.getByText(/Currently generating Fine-Tuned \+ RAG/)).toBeInTheDocument();
    });

    it('offers Prepare remaining answers when stopped, failed or partly ready', async () => {
      for (const state of ['partially_ready', 'stopped', 'failed'] as const) {
        await shown(eightyOfOneTwenty({ state }));
        expect(summaryButtons()).toEqual(['Prepare remaining answers']);
        cleanup();
      }
    });

    it('offers only Stop while preparing, and says stopping keeps the answers', async () => {
      await shown(RUNNING);
      expect(summaryButtons()).toEqual(['Stop preparation']);
      expect(screen.getByText(/Stopping keeps every answer prepared so far/)).toBeInTheDocument();

      api.stopPreparation.mockResolvedValue({
        ...RUNNING,
        state: 'stopping',
        stopping: true,
        message: 'Stopping after the current answer. Your completed answers are saved.',
      });
      fireEvent.click(screen.getByRole('button', { name: 'Stop preparation' }));

      expect(api.stopPreparation).toHaveBeenCalledWith(courseId);
      const stopping = await screen.findByRole('button', { name: 'Stopping…' });
      expect(stopping).toBeDisabled();
      expect(summaryButtons()).toEqual(['Stopping…']);
      expect(screen.getByText(/Your completed answers are saved/)).toBeInTheDocument();
    });

    it('offers nothing when every answer is ready', async () => {
      await shown(
        eightyOfOneTwenty({
          state: 'ready',
          canStart: false,
          startBlockedReason: 'nothing_to_prepare',
          counts: { ...eightyOfOneTwenty().counts, readyAnswers: 120, missingAnswers: 0, fullyPreparedQuestions: 30 },
          message: 'All 30 questions are ready for class.',
        }),
      );
      expect(screen.getByText('Ready')).toBeInTheDocument();
      expect(summaryButtons()).toEqual([]);
    });

    it('shows the server’s own sentence when a start is refused', async () => {
      await shown(preparation());
      api.startPreparation.mockRejectedValue(
        new ApiError(
          "Another course's answers are being prepared right now. Only one course is prepared at a time.",
          409,
          'preparation_another_course',
        ),
      );
      fireEvent.click(screen.getByRole('button', { name: 'Prepare answers' }));
      expect(await screen.findByText(/Only one course is prepared at a time/)).toBeInTheDocument();
      // Still usable afterwards, and re-read from the server.
      await waitFor(() => expect(api.fetchPreparationStatus).toHaveBeenCalledTimes(2));
      expect(summaryButtons()).toEqual(['Prepare answers']);
    });

    it('says something calm, not technical, when the server cannot be reached', async () => {
      await shown(preparation());
      api.startPreparation.mockRejectedValue(new ApiError('PostgreSQL is unavailable while starting.', 503));
      fireEvent.click(screen.getByRole('button', { name: 'Prepare answers' }));
      expect(await screen.findByText(/could not be reached just now/)).toBeInTheDocument();
      expect(screen.queryByText(/PostgreSQL/)).not.toBeInTheDocument();
    });
  });

  describe('progress without a refresh', () => {
    it('re-reads the status while a run is going and stops asking when it ends', async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      await shown(RUNNING);
      expect(api.fetchPreparationStatus).toHaveBeenCalledTimes(1);

      api.fetchPreparationStatus.mockResolvedValue({
        ...RUNNING,
        counts: { ...RUNNING.counts, readyAnswers: 81, missingAnswers: 39 },
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(RUNNING_POLL_MS);
      });
      expect(api.fetchPreparationStatus).toHaveBeenCalledTimes(2);
      expect(screen.getByText('Answers ready').closest('.prep-stat')).toHaveTextContent('81 of 120');

      api.fetchPreparationStatus.mockResolvedValue(
        eightyOfOneTwenty({
          state: 'stopped',
          message: 'Preparation stopped. Your completed answers are saved and students can use them now.',
        }),
      );
      await act(async () => {
        await vi.advanceTimersByTimeAsync(RUNNING_POLL_MS);
      });
      expect(screen.getByText('Stopped')).toBeInTheDocument();
      const calls = api.fetchPreparationStatus.mock.calls.length;
      await act(async () => {
        await vi.advanceTimersByTimeAsync(RUNNING_POLL_MS * 4);
      });
      expect(api.fetchPreparationStatus).toHaveBeenCalledTimes(calls);
    });

    it('does not poll at all when nothing is running', async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      await shown(eightyOfOneTwenty());
      await act(async () => {
        await vi.advanceTimersByTimeAsync(RUNNING_POLL_MS * 4);
      });
      expect(api.fetchPreparationStatus).toHaveBeenCalledTimes(1);
    });

    it('shows the server’s state when the page is opened again mid-run', async () => {
      await shown(RUNNING);
      cleanup();
      await shown(RUNNING);
      expect(screen.getByText('Preparing')).toBeInTheDocument();
      expect(summaryButtons()).toEqual(['Stop preparation']);
    });
  });

  describe('choosing questions', () => {
    it('lists the course questions with their source and current selection', async () => {
      await shown(preparation());
      const list = screen.getByRole('list', { name: 'Course questions' });
      const rows = within(list).getAllByRole('checkbox');
      // Behaviour examples are hidden until asked for.
      expect(rows.map((row) => (row as HTMLInputElement).checked)).toEqual([true, true, false]);
      expect(within(list).getAllByText('Staff-curated')).toHaveLength(2);
      expect(within(list).getByText('AI-generated')).toBeInTheDocument();
      expect(screen.queryByText('What is the weather today?')).not.toBeInTheDocument();
      expect(screen.getByText('3 of 4 questions shown · 2 selected')).toBeInTheDocument();
    });

    it('saves the whole selection as soon as a box is ticked', async () => {
      await shown(preparation());
      api.savePreparationSelection.mockResolvedValue(
        preparation({
          questions: preparation().questions.map((q) =>
            q.seedId === 's3'
              ? { ...q, selected: true, answers: { base: 'missing', rag: 'missing', fineTuned: 'missing', fineTunedRag: 'missing' } }
              : q,
          ),
        }),
      );
      fireEvent.click(screen.getByLabelText('How is the final grade calculated?'));

      expect(api.savePreparationSelection).toHaveBeenCalledWith(courseId, ['s1', 's2', 's3']);
      // Shown at once, before the server answers.
      expect(screen.getByLabelText('How is the final grade calculated?')).toBeChecked();
      await waitFor(() => expect(screen.getByText('3 of 4 questions shown · 3 selected')).toBeInTheDocument());
    });

    it('deselects a question the same way', async () => {
      await shown(preparation());
      api.savePreparationSelection.mockResolvedValue(preparation());
      fireEvent.click(screen.getByLabelText('When is the final exam?'));
      expect(api.savePreparationSelection).toHaveBeenCalledWith(courseId, ['s2']);
    });

    it('sends only the newest selection when boxes are ticked faster than the server answers', async () => {
      await shown(preparation());
      let release: (value: PreparationStatus) => void = () => undefined;
      api.savePreparationSelection.mockImplementationOnce(
        () =>
          new Promise<PreparationStatus>((resolve) => {
            release = resolve;
          }),
      );
      const withAll = preparation({
        questions: preparation().questions.map((q) => ({ ...q, selected: q.seedId !== 's4' })),
      });
      api.savePreparationSelection.mockResolvedValue(withAll);

      fireEvent.click(screen.getByLabelText('When is the final exam?')); // off -> [s2]
      fireEvent.click(screen.getByLabelText('How is the final grade calculated?')); // on -> [s2, s3]
      fireEvent.click(screen.getByLabelText('When is the final exam?')); // back on -> [s1, s2, s3]
      expect(api.savePreparationSelection).toHaveBeenCalledTimes(1);

      await act(async () => {
        release(preparation({ questions: preparation().questions.map((q) => ({ ...q, selected: q.seedId === 's2' })) }));
      });
      await waitFor(() => expect(api.savePreparationSelection).toHaveBeenCalledTimes(2));
      // The middle choice was never sent; the stale first answer never showed.
      expect(api.savePreparationSelection.mock.calls[1]).toEqual([courseId, ['s1', 's2', 's3']]);
      expect(screen.getByLabelText('When is the final exam?')).toBeChecked();
      expect(screen.getByLabelText('How is the final grade calculated?')).toBeChecked();
    });

    it('puts the boxes back and says so when a save is refused', async () => {
      await shown(preparation());
      api.savePreparationSelection.mockRejectedValue(
        new ApiError('A selected question cannot be prepared for the classroom.', 422),
      );
      fireEvent.click(screen.getByLabelText('How is the final grade calculated?'));
      expect(await screen.findByText('A selected question cannot be prepared for the classroom.')).toBeInTheDocument();
      await waitFor(() => expect(screen.getByLabelText('How is the final grade calculated?')).not.toBeChecked());
    });

    it('selects and clears everything shown, respecting the filters', async () => {
      await shown(preparation());
      // The server confirms each selection it is sent.
      api.savePreparationSelection.mockImplementation(async (_course: string, seedIds: string[]) =>
        preparation({
          questions: preparation().questions.map((q) => ({ ...q, selected: seedIds.includes(q.seedId) })),
        }),
      );

      fireEvent.change(screen.getByLabelText('Source'), { target: { value: 'ai_generated' } });
      expect(screen.getByText('1 of 4 questions shown · 2 selected')).toBeInTheDocument();
      fireEvent.click(screen.getByRole('button', { name: 'Select all shown' }));
      // Adds the one shown question; keeps the two already selected that are filtered out.
      expect(api.savePreparationSelection).toHaveBeenLastCalledWith(courseId, ['s1', 's2', 's3']);
      await waitFor(() => expect(screen.getByText('1 of 4 questions shown · 3 selected')).toBeInTheDocument());

      fireEvent.change(screen.getByLabelText('Source'), { target: { value: 'prototype' } });
      fireEvent.click(screen.getByRole('button', { name: 'Clear shown' }));
      expect(api.savePreparationSelection).toHaveBeenLastCalledWith(courseId, ['s3']);
    });

    it('searches, filters by source and can show only what is selected', async () => {
      await shown(preparation());
      const visible = () =>
        within(screen.getByRole('list', { name: 'Course questions' }))
          .getAllByRole('checkbox')
          .map((box) => box.id.split('-q-')[1]);

      fireEvent.change(screen.getByLabelText('Search questions'), { target: { value: 'LATE work' } });
      expect(visible()).toEqual(['s2']);
      fireEvent.change(screen.getByLabelText('Search questions'), { target: { value: '' } });

      fireEvent.change(screen.getByLabelText('Source'), { target: { value: 'authored' } });
      expect(visible()).toEqual(['s4']);
      expect(screen.getByText('Behaviour example')).toBeInTheDocument();
      fireEvent.change(screen.getByLabelText('Source'), { target: { value: 'all' } });

      fireEvent.click(screen.getByLabelText('Selected only'));
      expect(visible()).toEqual(['s1', 's2']);

      fireEvent.change(screen.getByLabelText('Search questions'), { target: { value: 'zzz' } });
      expect(screen.getByText('No matching questions')).toBeInTheDocument();
    });

    it('flags a selected question that can no longer be prepared and lets it be removed', async () => {
      await shown(
        preparation({
          questions: [
            ...preparation().questions.slice(0, 2),
            question('s9', 'A question rejected after it was chosen?', {
              selected: true,
              eligible: false,
              ineligibleReason: 'rejected',
              answers: null,
            }),
          ],
        }),
      );
      expect(screen.getByText(/Rejected in review, so it is no longer prepared/)).toBeInTheDocument();
      const box = screen.getByLabelText('A question rejected after it was chosen?');
      expect(box).toBeChecked();
      expect(box).toBeEnabled();
      api.savePreparationSelection.mockResolvedValue(preparation());
      fireEvent.click(box);
      expect(api.savePreparationSelection).toHaveBeenCalledWith(courseId, ['s1', 's2']);
    });

    it('says what choosing a question does and does not do', async () => {
      await shown(preparation());
      expect(
        screen.getByText(/does not suggest it to students and does not approve it for training/),
      ).toBeInTheDocument();
    });

    it('explains an empty course instead of showing an empty list', async () => {
      await shown(
        preparation({
          questions: [],
          canStart: false,
          counts: { ...preparation().counts, selectedQuestions: 0, expectedAnswers: 0 },
        }),
      );
      expect(screen.getByText('No questions to choose from yet')).toBeInTheDocument();
      expect(screen.queryByLabelText('Search questions')).not.toBeInTheDocument();
    });
  });

  describe('per-question readiness', () => {
    it('shows four labelled indicators for each selected question and none for the rest', async () => {
      await shown(
        preparation({
          questions: [
            question('s1', 'When is the final exam?', {
              selected: true,
              answers: { base: 'ready', rag: 'ready', fineTuned: 'missing', fineTunedRag: 'outdated' },
            }),
            question('s2', 'What is the late work policy?', {
              selected: true,
              answers: { base: 'ready', rag: 'unavailable', fineTuned: 'not_expected', fineTunedRag: 'not_expected' },
            }),
            question('s3', 'How is the final grade calculated?'),
          ],
        }),
      );
      const first = screen.getByRole('list', { name: 'Answers for: When is the final exam?' });
      expect(within(first).getAllByRole('listitem').map((item) => item.textContent)).toEqual([
        '✓BaseBase: ready',
        '✓RAGRAG: ready',
        '○FTFine-Tuned: not prepared yet',
        '↻FT+RAGFine-Tuned + RAG: out of date',
      ]);
      const second = screen.getByRole('list', { name: 'Answers for: What is the late work policy?' });
      expect(within(second).getByText('RAG: cannot be prepared right now')).toBeInTheDocument();
      expect(within(second).getAllByText(/not used by this course/)).toHaveLength(2);
      expect(
        screen.queryByRole('list', { name: 'Answers for: How is the final grade calculated?' }),
      ).not.toBeInTheDocument();
      // The legend says what each mark means.
      expect(screen.getByText('✓ ready')).toBeInTheDocument();
      expect(screen.getByText('↻ out of date')).toBeInTheDocument();
    });
  });

  describe('loading and failure', () => {
    it('says it is loading, then shows the page', async () => {
      let release: (value: PreparationStatus) => void = () => undefined;
      api.fetchPreparationStatus.mockReturnValue(
        new Promise<PreparationStatus>((resolve) => {
          release = resolve;
        }),
      );
      renderPage();
      expect(screen.getByText('Loading classroom answers…')).toBeInTheDocument();
      await act(async () => {
        release(preparation());
      });
      expect(await screen.findByRole('heading', { name: 'Preparation' })).toBeInTheDocument();
    });

    it('does not show zeroes when the status cannot be read, and retries on request', async () => {
      api.fetchPreparationStatus.mockRejectedValueOnce(new ApiError('boom', 503));
      renderPage();
      expect(await screen.findByText('Classroom answers unavailable')).toBeInTheDocument();
      expect(screen.getByText(/already prepared are not affected/)).toBeInTheDocument();
      expect(screen.queryByText('Answers ready')).not.toBeInTheDocument();

      api.fetchPreparationStatus.mockResolvedValue(eightyOfOneTwenty());
      fireEvent.click(screen.getByRole('button', { name: /try again/i }));
      expect(await screen.findByText('Partially ready')).toBeInTheDocument();
    });

    it('reads the course in the URL, and re-reads when it changes', async () => {
      await shown(preparation());
      expect(api.fetchPreparationStatus).toHaveBeenLastCalledWith('css-360-winter-2026-a7rp');
      cleanup();
      courseId = 'css-350-spring-2026-n3h9';
      await shown(preparation({ courseId }));
      expect(api.fetchPreparationStatus).toHaveBeenLastCalledWith('css-350-spring-2026-n3h9');
    });
  });

  describe('professor and administrator', () => {
    it('is the same page for both, with its own way back', async () => {
      await shown(eightyOfOneTwenty());
      const professorText = document.body.textContent?.replace('Back to course', '');
      expect(screen.getByRole('link', { name: 'Back to course' })).toHaveAttribute(
        'href',
        '/professor/course/css-360-winter-2026-a7rp',
      );
      expect(screen.queryByText('Admin')).not.toBeInTheDocument();
      cleanup();

      await shown(eightyOfOneTwenty(), 'admin');
      expect(screen.getByRole('link', { name: 'Back to course' })).toHaveAttribute(
        'href',
        '/admin/courses/css-360-winter-2026-a7rp',
      );
      expect(screen.getByText('Admin')).toBeInTheDocument();
      // Apart from the eyebrow, word for word the same.
      expect(document.body.textContent?.replace('Back to course', '').replace('Admin', '')).toBe(professorText);
    });
  });
});
