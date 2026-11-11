import type {
  AnswerState,
  ApproachKey,
  PreparationQuestion,
  PreparationStatus,
} from '../lib/classroomPreparation';

/** Fixtures for classroom-preparation tests: a status as the backend returns it. */

const ALL_READY: Record<ApproachKey, AnswerState> = {
  base: 'ready',
  rag: 'ready',
  fineTuned: 'ready',
  fineTunedRag: 'ready',
};

export function question(
  seedId: string,
  text: string,
  overrides: Partial<PreparationQuestion> = {},
): PreparationQuestion {
  const selected = overrides.selected ?? false;
  return {
    seedId,
    question: text,
    origin: 'prototype',
    source: 'prototype',
    sourceLabel: 'Staff-curated',
    reviewStatus: 'generated',
    category: 'general',
    selected,
    eligible: true,
    ineligibleReason: null,
    answers: selected ? { ...ALL_READY } : null,
    ...overrides,
  };
}

export function preparation(overrides: Partial<PreparationStatus> = {}): PreparationStatus {
  return {
    courseId: 'css-360-winter-2026-a7rp',
    enabled: true,
    state: 'not_started',
    running: false,
    stopping: false,
    waiting: null,
    current: null,
    canStart: true,
    startBlockedReason: null,
    counts: {
      selectedQuestions: 2,
      expectedAnswers: 8,
      readyAnswers: 0,
      missingAnswers: 8,
      outdatedAnswers: 0,
      unavailableAnswers: 0,
      fullyPreparedQuestions: 0,
      preparableAnswers: 8,
    },
    conditions: {
      base: { label: 'Base', expected: true, available: true, reason: null },
      rag: { label: 'RAG', expected: true, available: true, reason: null },
      fineTuned: { label: 'Fine-Tuned', expected: true, available: true, reason: null },
      fineTunedRag: { label: 'Fine-Tuned + RAG', expected: true, available: true, reason: null },
    },
    message: '2 questions selected. No answers are prepared yet.',
    notices: [],
    lastRun: null,
    questions: [
      question('s1', 'When is the final exam?', {
        selected: true,
        answers: { base: 'missing', rag: 'missing', fineTuned: 'missing', fineTunedRag: 'missing' },
      }),
      question('s2', 'What is the late work policy?', {
        selected: true,
        answers: { base: 'missing', rag: 'missing', fineTuned: 'missing', fineTunedRag: 'missing' },
      }),
      question('s3', 'How is the final grade calculated?', {
        source: 'ai_generated',
        origin: 'ai_generated',
        sourceLabel: 'AI-generated',
      }),
      question('s4', 'What is the weather today?', {
        source: 'authored',
        origin: 'authored',
        sourceLabel: 'Behaviour example',
      }),
    ],
    ...overrides,
  };
}

/** The example from the plan: 30 questions, 80 of 120 answers, 20 fully prepared. */
export function eightyOfOneTwenty(overrides: Partial<PreparationStatus> = {}): PreparationStatus {
  return preparation({
    state: 'partially_ready',
    counts: {
      selectedQuestions: 30,
      expectedAnswers: 120,
      readyAnswers: 80,
      missingAnswers: 40,
      outdatedAnswers: 0,
      unavailableAnswers: 0,
      fullyPreparedQuestions: 20,
      preparableAnswers: 40,
    },
    message: '20 of 30 questions fully ready. Students can already use every prepared answer.',
    ...overrides,
  });
}
