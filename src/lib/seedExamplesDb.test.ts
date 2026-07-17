import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { SeedExample } from '../types';

const { refMock, onValueMock, removeMock } = vi.hoisted(() => ({
  refMock: vi.fn((db: unknown, path: string) => ({ path, db })),
  onValueMock: vi.fn(),
  removeMock: vi.fn(async () => undefined),
}));

vi.mock('./firebase', () => ({
  app: {},
  database: { name: 'mock-db' },
}));

vi.mock('firebase/database', () => ({
  ref: refMock,
  onValue: onValueMock,
  push: vi.fn(() => ({ key: 'generated-id' })),
  set: vi.fn(async () => undefined),
  remove: removeMock,
  update: vi.fn(async () => undefined),
}));

import {
  deleteAllUserSeedExamples,
  parseSeedExamplesFromSnapshot,
  subscribeToSeedExamples,
} from './seedExamplesDb';

describe('seedExamplesDb AI-aware parsing', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('parses mixed user, prototype, and AI-generated snapshot records', () => {
    const seeds = parseSeedExamplesFromSnapshot({
      'user-1': {
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
      'ai-push': {
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
      malformed: {
        origin: 'ai_generated',
        question: '',
      },
    });

    expect(seeds).toHaveLength(2);
    expect(seeds.map((seed) => seed.origin).sort()).toEqual(['ai_generated', 'user']);
    expect(seeds[0].id).toBe('ai-push');
    expect(seeds.find((seed) => seed.origin === 'ai_generated')?.instruction).toBe(
      'AI question?',
    );
  });

  it('subscribes using the course-specific Firebase path', () => {
    onValueMock.mockReturnValue(() => undefined);

    subscribeToSeedExamples(
      'css-360-summer-2026-89m4',
      () => undefined,
      () => undefined,
    );

    expect(refMock).toHaveBeenCalledWith(
      expect.anything(),
      'courses/css-360-summer-2026-89m4/seedExamples',
    );
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

    expect(removeMock).toHaveBeenCalledTimes(1);
    expect(removeMock).toHaveBeenCalledWith({
      path: 'courses/css-360-summer-2026-89m4/seedExamples/user-1',
      db: expect.anything(),
    });
  });
});
