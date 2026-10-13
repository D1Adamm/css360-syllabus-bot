/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

/**
 * The Professor "Ask questions" tab: a course's staff comparing the four
 * approaches on their own question, with no class code.
 *
 * Held here: the four requests go out for the course in the URL, each card
 * reports its own answer or failure (and which model answered), and nothing
 * is kept that the study reads — no evaluation, no contribution, and no run
 * in the session storage the student Evaluate page reads from.
 */

const COURSE = vi.hoisted(() => 'css360e-autumn-2026-c08m');

const api = vi.hoisted(() => ({
  generateBaseModel: vi.fn(),
  generateRag: vi.fn(),
  generateFineTuned: vi.fn(),
  generateFineTunedRag: vi.fn(),
  listCourseSeeds: vi.fn(),
}));

vi.mock('../../lib/api', () => ({
  ApiError: class ApiError extends Error {},
  ...api,
}));

const research = vi.hoisted(() => ({
  createEvaluation: vi.fn(),
  previewEvaluation: vi.fn(),
  createSeed: vi.fn(),
}));

vi.mock('../../lib/evaluationsDb', () => ({
  createEvaluation: research.createEvaluation,
  previewEvaluation: research.previewEvaluation,
  subscribeToEvaluations: vi.fn(() => () => undefined),
  deleteEvaluation: vi.fn(),
  deleteAllEvaluations: vi.fn(),
}));

vi.mock('../../lib/seedExamplesDb', async () => {
  const actual = await vi.importActual<typeof import('../../lib/seedExamplesDb')>(
    '../../lib/seedExamplesDb',
  );
  return { ...actual, createSeedExample: research.createSeed };
});

import { CourseRoute } from '../../components/CourseRoute';
import { ApiError } from '../../lib/api';
import { SessionProvider, type Session } from '../../context/SessionContext';
import { ProfessorAskPage } from './ProfessorAskPage';

const PROFESSOR: Session = {
  user: { userId: 'u-prof', email: 'p@uw.edu', displayName: 'Prof', role: 'professor', courseIds: [COURSE] },
  participants: [],
};

const ADMIN: Session = {
  user: { userId: 'u-admin', email: 'a@uw.edu', displayName: 'Admin', role: 'admin', courseIds: [] },
  participants: [],
};

function renderAsk(session: Session = PROFESSOR) {
  return render(
    <MemoryRouter initialEntries={[`/professor/course/${COURSE}/ask`]}>
      <SessionProvider initialSession={session}>
        <Routes>
          <Route path="/professor/course/:courseId" element={<CourseRoute />}>
            <Route path="ask" element={<ProfessorAskPage />} />
          </Route>
        </Routes>
      </SessionProvider>
    </MemoryRouter>,
  );
}

function ask(question = 'When are office hours?') {
  fireEvent.change(screen.getByLabelText('What would you like to ask about this course?'), {
    target: { value: question },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Ask' }));
}

const MARKERS: Record<string, string> = { Base: 'A', RAG: 'B', 'Fine-Tuned': 'C', 'Fine-Tuned + RAG': 'D' };

function card(label: string): HTMLElement {
  const grid = screen.getByRole('region', { name: 'Responses' });
  const heading = within(grid)
    .getAllByRole('heading')
    .find((element) => element.textContent?.replace(/\s+/g, '') === `${MARKERS[label]}${label}`.replace(/\s+/g, ''));
  return heading?.closest('article') as HTMLElement;
}

beforeEach(() => {
  vi.clearAllMocks();
  window.sessionStorage.clear();
  api.listCourseSeeds.mockResolvedValue({ courseId: COURSE, count: 0, seeds: [] });
  api.generateBaseModel.mockResolvedValue({ answer: 'Base answer.', model: 'llama3.2:3b', responseType: 'base' });
  api.generateRag.mockResolvedValue({
    courseId: COURSE, answer: 'RAG answer.', model: 'llama3.2:3b', responseType: 'rag', retrievedChunks: [],
    sources: [{ chunkId: 'c1', sectionTitle: 'Office Hours', text: 'Tuesdays 2-3pm', score: 0.9 }],
  });
  api.generateFineTuned.mockResolvedValue({
    answer: 'Fine-tuned answer.', model: 'css360e-v1:latest', modelVersion: 'v1', responseType: 'fineTuned',
    adapterLoaded: true,
  });
  api.generateFineTunedRag.mockResolvedValue({
    courseId: COURSE, answer: 'Combined answer.', model: 'css360e-v1:latest', modelVersion: 'v1',
    responseType: 'fineTunedRag', adapterLoaded: true, sources: [], retrievedChunks: [],
  });
});

afterEach(() => {
  cleanup();
});

describe('ProfessorAskPage', () => {
  it('says the answers are not saved before anything is asked', async () => {
    renderAsk();
    expect(await screen.findByText('Staff testing — not saved')).toBeInTheDocument();
    expect(screen.getByText(/not added to this course's examples, student contributions or evaluation results/))
      .toBeInTheDocument();
  });

  it('asks all four approaches for this course, with no class code, and keeps nothing', async () => {
    renderAsk();
    ask();

    expect(await screen.findByText('Combined answer.')).toBeInTheDocument();
    await screen.findByText('Base answer.');
    await screen.findByText('RAG answer.');
    await screen.findByText('Fine-tuned answer.');
    for (const generate of Object.values(api).filter((fn) => fn !== api.listCourseSeeds)) {
      expect(generate).toHaveBeenCalledTimes(1);
      expect(generate.mock.calls[0][0]).toBe(COURSE);
      expect(generate.mock.calls[0][1]).toBe('When are office hours?');
    }

    // Which model and version answered, per card.
    expect(within(card('Fine-Tuned')).getByText('Answered by css360e-v1:latest · v1')).toBeInTheDocument();
    expect(within(card('Base')).getByText('Answered by llama3.2:3b')).toBeInTheDocument();

    // Sources are shown as on the student page.
    fireEvent.click(within(card('RAG')).getByRole('button', { name: /Where this came from/ }));
    expect(within(card('RAG')).getByText(/Office Hours/)).toBeInTheDocument();

    // Nothing for the study: no rating, no contribution, no run for Evaluate to find.
    expect(research.createEvaluation).not.toHaveBeenCalled();
    expect(research.previewEvaluation).not.toHaveBeenCalled();
    expect(research.createSeed).not.toHaveBeenCalled();
    expect(window.sessionStorage.length).toBe(0);
    expect(screen.queryByRole('link', { name: /Evaluate/ })).toBeNull();
  });

  it('reports a failed condition on its own card and keeps the other answers', async () => {
    api.generateFineTuned.mockRejectedValue(
      new ApiError('Fine-tuned inference service is unavailable. Check that the service is running.'),
    );
    renderAsk();
    ask();

    await screen.findByText('Combined answer.');
    await waitFor(() =>
      expect(within(card('Fine-Tuned')).getByText(/temporarily unavailable/)).toBeInTheDocument(),
    );
    // A professor is not shown infrastructure.
    expect(within(card('Fine-Tuned')).queryByText(/inference service/)).toBeNull();
    expect(within(card('Base')).getByText('Base answer.')).toBeInTheDocument();
  });

  it("gives an administrator the backend's own reason", async () => {
    api.generateFineTuned.mockRejectedValue(
      new ApiError('Fine-tuned inference service is unavailable. Check that the service is running.'),
    );
    renderAsk(ADMIN);
    ask();

    await waitFor(() =>
      expect(within(card('Fine-Tuned')).getByText(/inference service is unavailable/)).toBeInTheDocument(),
    );
  });

  it('sends one set of requests however often Ask is pressed', async () => {
    renderAsk();
    ask();
    fireEvent.click(screen.getByRole('button', { name: /Ask/ }));
    await screen.findByText('Combined answer.');
    expect(api.generateBaseModel).toHaveBeenCalledTimes(1);
    expect(api.generateFineTunedRag).toHaveBeenCalledTimes(1);
  });
});
