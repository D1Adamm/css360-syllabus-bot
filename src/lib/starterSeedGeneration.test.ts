import { describe, expect, it } from 'vitest';
import type { CourseMetadata } from '../types';
import {
  describeStarterGeneration,
  getStarterGeneration,
  isGeneratingStarterExamples,
  parseStarterGeneration,
} from './starterSeedGeneration';

const METADATA: CourseMetadata = {
  name: 'CSS 350',
  title: 'Management Principles',
  term: 'Winter 2026',
  instructorName: '',
  createdAt: '2026-08-12T09:00:00.000Z',
  syllabusStatus: 'indexed',
  syllabusFileName: 'syllabus.pdf',
  syllabusType: 'pdf',
  chunkCount: 12,
};

describe('parseStarterGeneration', () => {
  it('treats queued and generating as one wait', () => {
    expect(parseStarterGeneration({ status: 'queued' }).state).toBe('generating');
    expect(parseStarterGeneration({ status: 'generating' }).state).toBe('generating');
  });

  it('treats a partial run as ready, because there are examples to review', () => {
    expect(parseStarterGeneration({ status: 'ready' }).state).toBe('ready');
    expect(parseStarterGeneration({ status: 'partial' }).state).toBe('ready');
  });

  it('reads a failure', () => {
    expect(parseStarterGeneration({ status: 'failed' }).state).toBe('failed');
  });

  it('says not_started when nothing was ever recorded', () => {
    expect(parseStarterGeneration(undefined).state).toBe('not_started');
    expect(parseStarterGeneration(null).state).toBe('not_started');
    expect(parseStarterGeneration({}).state).toBe('not_started');
    expect(parseStarterGeneration({ status: 'not_started' }).state).toBe('not_started');
  });

  it('does not read an unknown status as a failure', () => {
    // A future job version writing a word this build has never seen must not
    // make a working course look broken.
    expect(parseStarterGeneration({ status: 'reticulating' }).state).toBe('not_started');
  });

  it('reports what was saved, not what was attempted', () => {
    const generation = parseStarterGeneration({
      status: 'ready',
      targetCount: 50,
      finalCount: 48,
      savedCount: 45,
      completedAt: '2026-08-12T10:30:00.000Z',
      startedAt: '2026-08-12T10:00:00.000Z',
    });

    expect(generation.generatedCount).toBe(45);
    expect(generation.startedAt).toBe('2026-08-12T10:00:00.000Z');
    expect(generation.completedAt).toBe('2026-08-12T10:30:00.000Z');
  });

  it('survives counts and timestamps it cannot use', () => {
    const generation = parseStarterGeneration({
      status: 'generating',
      savedCount: -4,
      startedAt: '',
    });

    expect(generation.generatedCount).toBe(0);
    expect(generation.startedAt).toBeNull();
  });
});

describe('getStarterGeneration', () => {
  it('reads the state from one course’s own metadata', () => {
    const metadata: CourseMetadata = {
      ...METADATA,
      starterSeedGeneration: { status: 'generating', targetCount: 50 },
    };

    expect(getStarterGeneration(metadata).state).toBe('generating');
    expect(isGeneratingStarterExamples(metadata)).toBe(true);
  });

  it('says nothing about a course whose metadata has no record', () => {
    expect(getStarterGeneration(METADATA).state).toBe('not_started');
    expect(isGeneratingStarterExamples(METADATA)).toBe(false);
    expect(isGeneratingStarterExamples(null)).toBe(false);
  });

  it('keeps courses isolated', () => {
    // The record is a field of one course's metadata, so a course with none
    // cannot inherit a neighbour's state.
    const busy: CourseMetadata = {
      ...METADATA,
      starterSeedGeneration: { status: 'generating' },
    };
    const quiet: CourseMetadata = { ...METADATA, name: 'CSS 360' };

    expect(isGeneratingStarterExamples(busy)).toBe(true);
    expect(isGeneratingStarterExamples(quiet)).toBe(false);
  });
});

describe('describeStarterGeneration', () => {
  it('tells a professor what is happening and how long it may take', () => {
    const message = describeStarterGeneration('generating');

    expect(message?.title).toBe('Generating starter examples…');
    expect(message?.detail).toMatch(/example questions from your syllabus/i);
    expect(message?.detail).toMatch(/several minutes/i);
  });

  it('never invents progress', () => {
    const message = describeStarterGeneration('generating');
    expect(`${message?.title} ${message?.detail}`).not.toMatch(/%|\d+ of \d+|percent/i);
  });

  it('says a failure safely, with no infrastructure in it', () => {
    const message = describeStarterGeneration('failed');

    expect(message?.title).toMatch(/couldn't create starter examples/i);
    expect(message?.detail).toMatch(/syllabus is saved/i);
    expect(message?.detail).toMatch(/administrator/i);
    expect(`${message?.title} ${message?.detail}`).not.toMatch(
      /ollama|model|token|timeout|http|firebase|traceback|exception|\.edu/i,
    );
  });

  it('carries no stored error text into professor wording', () => {
    // The record holds whatever the backend said; the wording is written here
    // and cannot include it.
    const generation = parseStarterGeneration({
      status: 'failed',
      error: 'ollama request to 127.0.0.1:11434 timed out after 600s',
    });
    const message = describeStarterGeneration(generation.state);

    expect(JSON.stringify(generation)).not.toMatch(/ollama|11434/);
    expect(`${message?.title} ${message?.detail}`).not.toMatch(/ollama|11434/);
  });

  it('adds nothing to the states that speak for themselves', () => {
    expect(describeStarterGeneration('ready')).toBeNull();
    expect(describeStarterGeneration('not_started')).toBeNull();
  });
});
