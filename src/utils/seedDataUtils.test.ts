import { describe, expect, it } from 'vitest';
import {
  getSeedOriginLabel,
  isSeedExample,
  normalizeSeedExample,
} from './seedDataUtils';

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
    expect(getSeedOriginLabel(seed!.origin)).toBe('AI-generated starter seed');
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
});
