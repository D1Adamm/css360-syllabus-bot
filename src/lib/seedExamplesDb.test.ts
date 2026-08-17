import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { SeedExample } from '../types';

const dbApiMock = vi.hoisted(() => ({
  listSeeds: vi.fn(),
  createSeed: vi.fn(),
  updateSeed: vi.fn(),
  deleteSeed: vi.fn(async () => ({ courseId: 'x', deleted: 1 })),
}));

vi.mock('./dbApi', () => dbApiMock);

import {
  deleteAllUserSeedExamples,
  parseSeedExampleList,
  subscribeToSeedExamples,
} from './seedExamplesDb';

describe('seedExamplesDb AI-aware parsing', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('parses mixed user, prototype, and AI-generated snapshot records', () => {
    const seeds = parseSeedExampleList([
      {
        id: 'user-1',
        instruction: 'User question?',
        response: 'User answer.',
        category: 'Course Basics',
        sourceSection: 'Meetings',
        difficulty: 'Easy',
        directlyAnswered: true,
        origin: 'user',
        createdAt: '2026-07-02T00:00:00.000Z',
      },
      {
        id: 'ai-push',
        question: 'AI question?',
        answer: 'AI answer with enough detail.',
        category: 'grading',
        sourceChunkIds: ['chunk-001'],
        sourceSection: 'Grading',
        origin: 'ai_generated',
        status: 'generated',
        createdAt: '2026-07-03T00:00:00.000Z',
        validation: {
          grounded: true,
          correct: true,
          clear: true,
          useful: true,
          score: 0.9,
          reason: 'Grounded.',
        },
      },
      {
        id: 'malformed',
        origin: 'ai_generated',
        question: '',
      },
    ] as never);

    expect(seeds).toHaveLength(2);
    expect(seeds.map((seed) => seed.origin).sort()).toEqual(['ai_generated', 'user']);
    expect(seeds[0].id).toBe('ai-push');
    expect(seeds.find((seed) => seed.origin === 'ai_generated')?.instruction).toBe(
      'AI question?',
    );
  });

  it('lists seeds for the course it was given', async () => {
    dbApiMock.listSeeds.mockResolvedValue({
      courseId: 'css-360-summer-2026-89m4',
      count: 0,
      seeds: [],
      reviewStatusCounts: {},
    });

    const unsubscribe = subscribeToSeedExamples(
      'css-360-summer-2026-89m4',
      () => undefined,
      () => undefined,
    );
    await new Promise((resolve) => setTimeout(resolve, 0));
    unsubscribe();

    expect(dbApiMock.listSeeds).toHaveBeenCalledWith('css-360-summer-2026-89m4');
  });

  it('deleteAllUserSeedExamples deletes only user seeds', async () => {
    const seeds: SeedExample[] = [
      {
        id: 'user-1',
        instruction: 'User Q?',
        response: 'User A.',
        category: 'Course Basics',
        sourceSection: 'Meetings',
        difficulty: 'Easy',
        directlyAnswered: true,
        origin: 'user',
      },
      {
        id: 'ai-1',
        instruction: 'AI Q?',
        response: 'AI A.',
        category: 'grading',
        sourceSection: 'Grading',
        difficulty: 'Medium',
        directlyAnswered: true,
        origin: 'ai_generated',
      },
      {
        id: 'proto-1',
        instruction: 'Prototype Q?',
        response: 'Prototype A.',
        category: 'Course Basics',
        sourceSection: 'Meetings',
        difficulty: 'Easy',
        directlyAnswered: true,
        origin: 'prototype',
      },
    ];

    await deleteAllUserSeedExamples('css-360-summer-2026-89m4', seeds);

    // Only the student's own example goes, and it is deleted by course and id
    // together — an id alone is never enough to reach another course's record.
    expect(dbApiMock.deleteSeed).toHaveBeenCalledTimes(1);
    expect(dbApiMock.deleteSeed).toHaveBeenCalledWith(
      'css-360-summer-2026-89m4',
      'user-1',
    );
  });
});
