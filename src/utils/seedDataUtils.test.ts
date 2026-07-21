import { describe, expect, it } from 'vitest';
import {
  calculateStatistics,
  filterSeeds,
  getSeedOriginLabel,
  isSeedExample,
  normalizeSeedExample,
  resolveSeedReviewStatus,
} from './seedDataUtils';
import type { SeedExample } from '../types';

describe('normalizeSeedExample', () => {
  it('parses AI-generated records with dual field names', () => {
    const seed = normalizeSeedExample(
      {
        question: 'When are office hours?',
        instruction: 'When are office hours?',
        answer: 'Office hours are on Tuesdays at 2pm.',
        response: 'Office hours are on Tuesdays at 2pm.',
        category: 'office hours',
        sourceChunkIds: ['chunk-001'],
        sourceSection: 'Office Hours',
        difficulty: 'Medium',
        directlyAnswered: true,
        origin: 'ai_generated',
        status: 'generated',
        createdAt: '2026-07-16T12:00:00.000Z',
        validation: {
          grounded: true,
          correct: true,
          clear: true,
          useful: true,
          score: 0.95,
          reason: 'Supported by the source chunk.',
        },
      },
      'push-ai-1',
    );

    expect(seed).not.toBeNull();
    expect(seed?.id).toBe('push-ai-1');
    expect(seed?.instruction).toBe('When are office hours?');
    expect(seed?.response).toBe('Office hours are on Tuesdays at 2pm.');
    expect(seed?.origin).toBe('ai_generated');
    expect(seed?.status).toBe('generated');
    expect(seed?.sourceChunkIds).toEqual(['chunk-001']);
    expect(seed?.validation?.score).toBe(0.95);
    expect(seed?.validation?.grounded).toBe(true);
    expect(getSeedOriginLabel(seed!.origin)).toBe('AI-generated starter seed');
  });

  it('parses new rubric validation records with components and unsupportedClaims', () => {
    const seed = normalizeSeedExample(
      {
        question: 'How should we organize project work?',
        answer: 'Team projects require weekly status updates and a final demo.',
        category: 'Team Project Requirements',
        questionType: 'scenario',
        sourceChunkIds: ['chunk-006'],
        origin: 'ai_generated',
        status: 'generated',
        validation: {
          score: 0.84,
          reason: 'Grounded but one embellished claim.',
          unsupportedClaims: ['Suggested coding/docs/testing split'],
          components: {
            grounded: 0.78,
            correct: 0.8,
            clear: 0.88,
            useful: 0.9,
            naturalStudentWording: 0.86,
            categoryCorrect: 0.85,
            notTrivialOrTemporary: 0.82,
          },
        },
      },
      'push-ai-2',
    );

    expect(seed).not.toBeNull();
    expect(seed?.questionType).toBe('scenario');
    expect(seed?.validation?.score).toBe(0.84);
    expect(seed?.validation?.components?.grounded).toBe(0.78);
    expect(seed?.validation?.unsupportedClaims).toEqual([
      'Suggested coding/docs/testing split',
    ]);
  });

  it('parses AI records that only provide question/answer', () => {
    const seed = normalizeSeedExample(
      {
        question: 'Can I submit late work?',
        answer: 'Late work may be submitted within 24 hours for half credit.',
        category: 'late policy',
        sourceChunkIds: ['chunk-004'],
        origin: 'ai_generated',
        status: 'generated',
      },
      'ai-only-fields',
    );

    expect(seed).not.toBeNull();
    expect(seed?.instruction).toBe('Can I submit late work?');
    expect(seed?.response).toContain('half credit');
    expect(seed?.sourceSection).toBe('chunk-004');
    expect(seed?.difficulty).toBe('Medium');
    expect(seed?.directlyAnswered).toBe(true);
  });

  it('parses existing user records', () => {
    const seed = normalizeSeedExample({
      id: 'user-1',
      instruction: 'When does class meet?',
      response: 'Tuesday and Thursday.',
      category: 'Course Basics',
      sourceSection: 'Course Meetings',
      difficulty: 'Easy',
      directlyAnswered: true,
      origin: 'user',
    });

    expect(seed).not.toBeNull();
    expect(seed?.origin).toBe('user');
    expect(isSeedExample(seed)).toBe(true);
  });

  it('parses existing prototype records', () => {
    const seed = normalizeSeedExample({
      id: 'proto-1',
      instruction: 'What is the AI policy?',
      response: 'Use AI tools only as permitted by the syllabus.',
      category: 'AI Policy',
      sourceSection: 'Academic Integrity',
      difficulty: 'Medium',
      directlyAnswered: true,
      origin: 'prototype',
    });

    expect(seed).not.toBeNull();
    expect(seed?.origin).toBe('prototype');
    expect(getSeedOriginLabel(seed!.origin)).toBe('Prototype generated');
  });

  it('rejects malformed records', () => {
    expect(normalizeSeedExample(null)).toBeNull();
    expect(normalizeSeedExample({ origin: 'user' })).toBeNull();
    expect(
      normalizeSeedExample({
        id: 'bad',
        instruction: 'Question?',
        response: 'Answer.',
        category: 'General',
        sourceSection: 'General',
        difficulty: 'Easy',
        directlyAnswered: true,
        origin: 'robot',
      }),
    ).toBeNull();
    expect(
      normalizeSeedExample({
        question: '',
        answer: 'Only an answer',
        category: 'general',
        origin: 'ai_generated',
      }),
    ).toBeNull();
  });

  it('preserves reviewStatus and edit provenance fields', () => {
    const seed = normalizeSeedExample(
      {
        question: 'Edited question?',
        answer: 'Edited answer with enough detail.',
        category: 'Late Work',
        origin: 'ai_generated',
        reviewStatus: 'approved',
        status: 'approved',
        wasEdited: true,
        originalQuestion: 'Original question?',
        originalAnswer: 'Original answer.',
        reviewNotes: 'Wording tweak',
      },
      'seed-reviewed',
    );

    expect(seed?.reviewStatus).toBe('approved');
    expect(seed?.wasEdited).toBe(true);
    expect(seed?.originalQuestion).toBe('Original question?');
    expect(seed?.originalAnswer).toBe('Original answer.');
    expect(seed?.reviewNotes).toBe('Wording tweak');
  });
});

describe('dataset review status helpers', () => {
  const seeds: SeedExample[] = [
    {
      id: 'a',
      instruction: 'Approved Q?',
      response: 'Approved A.',
      category: 'Grading',
      sourceSection: 'Grading',
      difficulty: 'Easy',
      directlyAnswered: true,
      origin: 'ai_generated',
      reviewStatus: 'approved',
    },
    {
      id: 'b',
      instruction: 'Generated Q?',
      response: 'Generated A.',
      category: 'Grading',
      sourceSection: 'Grading',
      difficulty: 'Easy',
      directlyAnswered: true,
      origin: 'ai_generated',
      reviewStatus: 'generated',
    },
    {
      id: 'c',
      instruction: 'Rejected Q?',
      response: 'Rejected A.',
      category: 'Late Work',
      sourceSection: 'Late Work',
      difficulty: 'Medium',
      directlyAnswered: true,
      origin: 'ai_generated',
      status: 'rejected',
    },
  ];

  it('resolves reviewStatus with status fallback', () => {
    expect(resolveSeedReviewStatus(seeds[0])).toBe('approved');
    expect(resolveSeedReviewStatus(seeds[2])).toBe('rejected');
    expect(
      resolveSeedReviewStatus({
        ...seeds[0],
        reviewStatus: undefined,
        status: undefined,
      }),
    ).toBe('generated');
  });

  it('counts review statuses for dataset summary', () => {
    const stats = calculateStatistics(seeds);
    expect(stats.totalExamples).toBe(3);
    expect(stats.approvedCount).toBe(1);
    expect(stats.generatedCount).toBe(1);
    expect(stats.rejectedCount).toBe(1);
    expect(stats.editedCount).toBe(0);
  });

  it('defaults filterSeeds reviewStatus to all when omitted', () => {
    expect(
      filterSeeds(seeds, {
        searchQuery: '',
        category: 'All categories',
        difficulty: 'All difficulties',
        answerType: 'All',
        sortBy: 'id-asc',
      }),
    ).toHaveLength(3);
  });

  it('filters to approved only when requested', () => {
    const filtered = filterSeeds(seeds, {
      searchQuery: '',
      category: 'All categories',
      difficulty: 'All difficulties',
      answerType: 'All',
      reviewStatus: 'approved',
      sortBy: 'id-asc',
    });
    expect(filtered).toHaveLength(1);
    expect(filtered[0].id).toBe('a');
  });
});
