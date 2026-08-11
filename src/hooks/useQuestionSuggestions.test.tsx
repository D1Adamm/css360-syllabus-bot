/** @vitest-environment jsdom */
import { renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const listCourseSeeds = vi.fn();

vi.mock('../lib/api', () => ({
  listCourseSeeds: (...args: unknown[]) => listCourseSeeds(...args),
}));

import comparisonData from '../data/comparisonData.json';
import { GENERIC_SUGGESTIONS, useQuestionSuggestions } from './useQuestionSuggestions';

const BUNDLED_QUESTIONS = (comparisonData as { question: string }[]).map(
  (record) => record.question,
);

function seed(id: string, question: string, reviewStatus: string) {
  return { id, question, answer: 'An answer.', reviewStatus };
}

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('useQuestionSuggestions', () => {
  it('requests suggestions for the course it was given', async () => {
    listCourseSeeds.mockResolvedValue({ seeds: [] });

    renderHook(() => useQuestionSuggestions('css-490-spring-2026-cgvl'));

    await waitFor(() => {
      expect(listCourseSeeds).toHaveBeenCalledWith('css-490-spring-2026-cgvl');
    });
  });

  it('suggests the course’s own approved examples', async () => {
    listCourseSeeds.mockResolvedValue({
      seeds: [
        seed('1', 'How do we pick an open source project?', 'approved'),
        seed('2', 'What counts as a meaningful contribution?', 'edited'),
        seed('3', 'Not reviewed yet', 'generated'),
        seed('4', 'Was rejected', 'rejected'),
      ],
    });

    const { result } = renderHook(() =>
      useQuestionSuggestions('css-490-spring-2026-cgvl'),
    );

    await waitFor(() => {
      expect(result.current.source).toBe('course');
    });

    expect(result.current.questions).toEqual([
      'How do we pick an open source project?',
      'What counts as a meaningful contribution?',
    ]);
  });

  it('never suggests another course’s bundled example questions', async () => {
    listCourseSeeds.mockResolvedValue({
      seeds: [seed('1', 'How do we pick an open source project?', 'approved')],
    });

    const { result } = renderHook(() =>
      useQuestionSuggestions('css-490-spring-2026-cgvl'),
    );

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // The bundled file was written for one specific course; several of its
    // entries assume that course's practices and are meaningless elsewhere.
    for (const question of result.current.questions) {
      expect(BUNDLED_QUESTIONS).not.toContain(question);
    }
  });

  it('falls back to course-agnostic questions when a course has none approved', async () => {
    listCourseSeeds.mockResolvedValue({
      seeds: [seed('1', 'Still awaiting review', 'generated')],
    });

    const { result } = renderHook(() => useQuestionSuggestions('new-course-abcd'));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.source).toBe('generic');
    expect(result.current.questions).toEqual([...GENERIC_SUGGESTIONS]);
  });

  it('falls back quietly when the request fails', async () => {
    listCourseSeeds.mockRejectedValue(new Error('offline'));

    const { result } = renderHook(() => useQuestionSuggestions('any-course-abcd'));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Suggestions are a convenience; a failure must not block the question box.
    expect(result.current.source).toBe('generic');
    expect(result.current.questions.length).toBeGreaterThan(0);
  });

  it('drops duplicates and caps the list', async () => {
    listCourseSeeds.mockResolvedValue({
      seeds: [
        seed('1', 'Repeated question', 'approved'),
        seed('2', 'repeated QUESTION', 'approved'),
        ...Array.from({ length: 10 }, (_, index) =>
          seed(`x${index}`, `Question ${index}`, 'approved'),
        ),
      ],
    });

    const { result } = renderHook(() => useQuestionSuggestions('any-course-abcd'));

    await waitFor(() => {
      expect(result.current.source).toBe('course');
    });

    expect(result.current.questions).toHaveLength(6);
    expect(result.current.questions.filter((q) => /repeated/i.test(q))).toHaveLength(1);
  });

  it('every generic fallback is answerable for any course', () => {
    // No course-specific practice may be assumed: no standups, no Discord, no
    // "final reflection", no "open lab".
    for (const question of GENERIC_SUGGESTIONS) {
      expect(question).not.toMatch(
        /standup|discord|open lab|final reflection|in-class activity|project task/i,
      );
    }
  });
});
