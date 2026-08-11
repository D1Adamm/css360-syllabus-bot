/** @vitest-environment jsdom */
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import type { SeedExample } from '../../types';

const addSeedMock = vi.fn();
const deleteSeedMock = vi.fn();
let seeds: SeedExample[] = [];

vi.mock('../../hooks/useSeedExamples', () => ({
  useSeedExamples: () => ({
    seeds,
    loading: false,
    error: null,
    saving: false,
    saveError: null,
    addSeed: addSeedMock,
    deleteSeed: deleteSeedMock,
    deleteAllSeeds: vi.fn(),
    clearSaveError: vi.fn(),
  }),
}));

vi.mock('../../context/CourseContext', () => ({
  useCourseId: () => 'css-360-winter-2026-a7rp',
}));

import { ContributePage } from './ContributePage';

const QUESTION = 'How much of my grade comes from the final project?';
const ANSWER = 'The final project is worth 30% of the overall course grade.';

function fillAndSubmit(question = QUESTION, answer = ANSWER) {
  fireEvent.change(screen.getByLabelText(/Your question/), {
    target: { value: question },
  });
  fireEvent.change(screen.getByLabelText(/The answer you would expect/), {
    target: { value: answer },
  });
  fireEvent.click(screen.getByRole('button', { name: /Add question/ }));
}

beforeEach(() => {
  vi.clearAllMocks();
  window.sessionStorage.clear();
  seeds = [];
  addSeedMock.mockResolvedValue(undefined);
});

afterEach(() => {
  cleanup();
});

describe('ContributePage', () => {
  it('never asks a student for their name', () => {
    render(<ContributePage />);

    expect(screen.queryByLabelText(/name/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Your name is not attached/i)).toBeInTheDocument();
  });

  it('uses no research or pipeline vocabulary', () => {
    render(<ContributePage />);

    expect(document.body.textContent).not.toMatch(
      /seed|dataset|fine-tun|validation score|difficulty/i,
    );
  });

  it('requires a question and an answer', async () => {
    render(<ContributePage />);

    fireEvent.click(screen.getByRole('button', { name: /Add question/ }));

    expect(await screen.findByText('Enter a question.')).toBeInTheDocument();
    expect(
      screen.getByText('Enter the answer you would expect.'),
    ).toBeInTheDocument();
    expect(addSeedMock).not.toHaveBeenCalled();
  });

  it('saves a record in the existing stored shape', async () => {
    render(<ContributePage />);
    fillAndSubmit();

    await waitFor(() => {
      expect(addSeedMock).toHaveBeenCalledTimes(1);
    });

    const saved = addSeedMock.mock.calls[0][0] as SeedExample;
    expect(saved.instruction).toBe(QUESTION);
    expect(saved.response).toBe(ANSWER);
    // origin drives the existing delete-all-user behaviour; it must not change.
    expect(saved.origin).toBe('user');
    // Fields the student no longer sets still carry safe defaults.
    expect(saved.difficulty).toBe('Medium');
    expect(saved.directlyAnswered).toBe(true);
    expect(saved.category).toBe('General');
    expect(saved.sourceSection).toBe('Not specified');
    expect(typeof saved.id).toBe('string');
    // No identity is attached to a contribution.
    expect(Object.keys(saved)).not.toContain('author');
    expect(Object.keys(saved)).not.toContain('participantId');
  });

  it('rejects a question that has already been contributed', async () => {
    seeds = [
      {
        id: 'seed-1',
        instruction: QUESTION,
        response: 'Existing answer that is long enough to be valid.',
        category: 'Grading',
        sourceSection: 'Grading',
        difficulty: 'Medium',
        directlyAnswered: true,
        origin: 'user',
      },
    ];

    render(<ContributePage />);
    fillAndSubmit();

    expect(
      await screen.findByText('Someone has already added this question.'),
    ).toBeInTheDocument();
    expect(addSeedMock).not.toHaveBeenCalled();
  });

  it('does not list other students\u2019 contributions', () => {
    seeds = [
      {
        id: 'someone-else',
        instruction: 'A classmate question that must stay private',
        response: 'An answer another student wrote for this course.',
        category: 'Policies',
        sourceSection: 'Late work',
        difficulty: 'Medium',
        directlyAnswered: true,
        origin: 'user',
      },
    ];

    render(<ContributePage />);

    expect(
      screen.queryByText('A classmate question that must stay private'),
    ).not.toBeInTheDocument();
    expect(screen.getByText('Nothing added yet')).toBeInTheDocument();
  });

  it('lists what this session added and confirms before removing one', async () => {
    seeds = [
      {
        id: 'seed-1',
        instruction: 'What is the late policy?',
        response: 'Late work may be submitted within 24 hours.',
        category: 'Policies',
        sourceSection: 'Late work',
        difficulty: 'Medium',
        directlyAnswered: true,
        origin: 'user',
      },
    ];
    window.sessionStorage.setItem(
      'sml.contributions.css-360-winter-2026-a7rp',
      JSON.stringify(['seed-1']),
    );

    render(<ContributePage />);

    expect(screen.getByText('What is the late policy?')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Remove/ }));

    const dialog = await screen.findByRole('alertdialog');
    expect(dialog).toHaveAccessibleName('Remove this question?');
    expect(deleteSeedMock).not.toHaveBeenCalled();

    fireEvent.click(within(dialog).getByRole('button', { name: 'Remove' }));
    await waitFor(() => {
      expect(deleteSeedMock).toHaveBeenCalledWith('seed-1');
    });
  });

  it('handles an older record that has no section recorded', () => {
    window.sessionStorage.setItem(
      'sml.contributions.css-360-winter-2026-a7rp',
      JSON.stringify(['seed-legacy']),
    );
    seeds = [
      {
        id: 'seed-legacy',
        instruction: 'Older question with no section',
        response: 'An answer stored before sections were captured.',
        category: 'General',
        sourceSection: 'Not specified',
        difficulty: 'Medium',
        directlyAnswered: true,
        origin: 'user',
      },
    ];

    render(<ContributePage />);

    expect(screen.getByText('Older question with no section')).toBeInTheDocument();
    expect(screen.queryByText('Not specified')).not.toBeInTheDocument();
  });
});
