/** @vitest-environment jsdom */
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Link, MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { ComparisonRunProvider } from '../../context/ComparisonRunContext';
import { CourseProvider } from '../../context/CourseContext';
import {
  ApiError,
  generateBaseModel,
  generateFineTuned,
  generateFineTunedRag,
  generateRag,
} from '../../lib/api';
import { ComparePage } from './ComparePage';

/*
 * A comparison belongs to the application, not to the Compare page: a student
 * asks, goes to Contribute while the four answers generate, comes back, and
 * finds the same question with whatever has arrived — nothing re-sent,
 * nothing lost, each condition settling on its own.
 */

vi.mock('../../lib/api', () => ({
  ApiError: class ApiError extends Error {
    status?: number;
    constructor(message: string, status?: number) {
      super(message);
      this.name = 'ApiError';
      this.status = status;
    }
  },
  generateBaseModel: vi.fn(),
  generateFineTuned: vi.fn(),
  generateFineTunedRag: vi.fn(),
  generateRag: vi.fn(),
  listCourseSeeds: vi.fn(async () => ({ courseId: 'c', count: 0, seeds: [] })),
}));

const COURSE = 'css-360-fall-2026-k7q2';
const QUESTION = 'When is the class?';

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const base = vi.mocked(generateBaseModel);
const rag = vi.mocked(generateRag);
const fineTuned = vi.mocked(generateFineTuned);
const fineTunedRag = vi.mocked(generateFineTunedRag);

function renderApp() {
  const path = (segment: string) => `/student/course/${COURSE}/${segment}`;
  return render(
    <MemoryRouter initialEntries={[path('compare')]}>
      <ComparisonRunProvider>
        <nav>
          <Link to={path('compare')}>Go to Compare</Link>
          <Link to={path('contribute')}>Go to Contribute</Link>
          <Link to={`/student/course/${COURSE}`}>Go Home</Link>
        </nav>
        <Routes>
          <Route
            path="/student/course/:courseId/compare"
            element={
              <CourseProvider courseId={COURSE}>
                <ComparePage />
              </CourseProvider>
            }
          />
          <Route path="/student/course/:courseId/contribute" element={<p>Contribute page</p>} />
          <Route path="/student/course/:courseId" element={<p>Course home</p>} />
        </Routes>
      </ComparisonRunProvider>
    </MemoryRouter>,
  );
}

function ask(question = QUESTION) {
  fireEvent.change(screen.getByLabelText('What would you like to ask about this course?'), {
    target: { value: question },
  });
  fireEvent.click(screen.getByRole('button', { name: /^(Ask|Asking…)$/ }));
}

function goTo(label: string) {
  fireEvent.click(screen.getByRole('link', { name: label }));
}

describe('a comparison survives leaving the Compare page', () => {
  let pending: {
    base: ReturnType<typeof deferred<Awaited<ReturnType<typeof generateBaseModel>>>>;
    rag: ReturnType<typeof deferred<Awaited<ReturnType<typeof generateRag>>>>;
    fineTuned: ReturnType<typeof deferred<Awaited<ReturnType<typeof generateFineTuned>>>>;
    fineTunedRag: ReturnType<typeof deferred<Awaited<ReturnType<typeof generateFineTunedRag>>>>;
  };

  beforeEach(() => {
    window.sessionStorage.clear();
    pending = {
      base: deferred(),
      rag: deferred(),
      fineTuned: deferred(),
      fineTunedRag: deferred(),
    };
    base.mockReset().mockReturnValue(pending.base.promise);
    rag.mockReset().mockReturnValue(pending.rag.promise);
    fineTuned.mockReset().mockReturnValue(pending.fineTuned.promise);
    fineTunedRag.mockReset().mockReturnValue(pending.fineTunedRag.promise);
  });

  afterEach(() => {
    cleanup();
  });

  it('keeps the question, finished answers, failures and pending cards across navigation', async () => {
    renderApp();
    ask();
    expect(await screen.findByText(QUESTION)).toBeInTheDocument();

    // One condition finishes while the student is still watching: shown at once.
    await act(async () => {
      pending.fineTuned.resolve({
        answer: 'Tuesdays and Thursdays at 3:30.',
        model: 'css360-ft-v2',
        responseType: 'fineTuned',
        courseId: COURSE,
        adapterLoaded: true,
        generationSeconds: 1,
      });
    });
    expect(screen.getByText('Tuesdays and Thursdays at 3:30.')).toBeInTheDocument();
    expect(screen.getByText('Waiting for all four responses…')).toBeInTheDocument();

    // Off to Contribute: the page unmounts, the requests do not.
    goTo('Go to Contribute');
    expect(screen.getByText('Contribute page')).toBeInTheDocument();
    expect(screen.queryByText(QUESTION)).not.toBeInTheDocument();

    // Base fails and RAG answers while the student is away.
    await act(async () => {
      pending.base.reject(new ApiError('Ollama request timed out.', 503));
    });
    await waitFor(() => expect(rag).toHaveBeenCalledTimes(1));
    await act(async () => {
      pending.rag.resolve({
        courseId: COURSE,
        answer: 'Class meets Tuesdays and Thursdays, 3:30 to 5:30 pm.',
        model: 'llama3.2:3b',
        responseType: 'rag',
        sources: [],
        retrievedChunks: [],
      });
    });

    goTo('Go Home');
    goTo('Go to Compare');

    // The same comparison: its question, both answers, the failure on its own
    // card, and the one still generating.
    expect(screen.getByText(QUESTION)).toBeInTheDocument();
    expect(screen.getByText('Tuesdays and Thursdays at 3:30.')).toBeInTheDocument();
    expect(
      screen.getByText('Class meets Tuesdays and Thursdays, 3:30 to 5:30 pm.'),
    ).toBeInTheDocument();
    expect(screen.getByText(/temporarily unavailable/)).toBeInTheDocument();
    expect(screen.getAllByText('Generating a response…')).toHaveLength(1);
    expect(screen.getByRole('button', { name: 'Asking…' })).toBeDisabled();

    // Coming back re-sent nothing.
    expect(base).toHaveBeenCalledTimes(1);
    expect(rag).toHaveBeenCalledTimes(1);
    expect(fineTuned).toHaveBeenCalledTimes(1);
    expect(fineTunedRag).toHaveBeenCalledTimes(1);

    // The last condition lands, and only then is the set ready to evaluate.
    await act(async () => {
      pending.fineTunedRag.resolve({
        courseId: COURSE,
        answer: 'The syllabus says Tuesdays and Thursdays.',
        model: 'css360-ft-v2',
        responseType: 'fineTunedRag',
        adapterLoaded: true,
        generationSeconds: 2,
        sources: [],
        retrievedChunks: [],
      });
    });
    expect(
      await screen.findByRole('link', { name: 'Evaluate these responses' }),
    ).toBeInTheDocument();
    const stored = JSON.parse(window.sessionStorage.getItem(`sml.run.${COURSE}`) ?? 'null');
    expect(stored.question).toBe(QUESTION);
    expect(stored.responses.base.error).toMatch(/temporarily unavailable/);
    expect(stored.responses.rag.text).toMatch(/3:30 to 5:30/);
    expect(stored.responses.fineTunedRag.text).toMatch(/Tuesdays and Thursdays/);
  });

  it('finishes while the student is away and is complete on return', async () => {
    renderApp();
    ask();
    goTo('Go to Contribute');

    await act(async () => {
      pending.base.resolve({ answer: 'I have no syllabus.', model: 'm', responseType: 'base', courseId: COURSE });
      pending.fineTuned.reject(new ApiError('down', 503));
      pending.fineTunedRag.reject(new ApiError('down', 503));
    });
    await waitFor(() => expect(rag).toHaveBeenCalledTimes(1));
    await act(async () => {
      pending.rag.resolve({
        courseId: COURSE,
        answer: 'Tuesdays.',
        model: 'm',
        responseType: 'rag',
        sources: [],
        retrievedChunks: [],
      });
    });
    await waitFor(() =>
      expect(window.sessionStorage.getItem(`sml.run.${COURSE}`)).not.toBeNull(),
    );

    goTo('Go to Compare');
    expect(screen.getByText(QUESTION)).toBeInTheDocument();
    expect(screen.getByText('I have no syllabus.')).toBeInTheDocument();
    expect(screen.getByText('Tuesdays.')).toBeInTheDocument();
    expect(screen.getAllByText(/temporarily unavailable/)).toHaveLength(2);
    expect(screen.getByRole('link', { name: 'Evaluate these responses' })).toBeInTheDocument();
    // Settled: the lock is released, so the next question can be asked.
    expect(screen.queryByRole('button', { name: 'Asking…' })).not.toBeInTheDocument();
    ask('Is there a final exam?');
    expect(base).toHaveBeenCalledTimes(2);
  });

  it('refuses a second comparison while one is running, even from a fresh page', async () => {
    renderApp();
    ask();
    goTo('Go to Contribute');
    goTo('Go to Compare');

    // The remounted page starts disabled; even forcing a submit sends nothing.
    ask('A different question?');
    expect(base).toHaveBeenCalledTimes(1);
    expect(fineTuned).toHaveBeenCalledTimes(1);
    expect(screen.getByText(QUESTION)).toBeInTheDocument();
    expect(screen.queryByText('A different question?')).not.toBeInTheDocument();
  });
});
