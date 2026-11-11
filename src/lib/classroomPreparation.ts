import { getJson, sendJson } from './httpClient';
import type { StatusTone } from '../components/ui/StatusPill';

/**
 * Classroom preparation for one course: which existing course questions staff
 * selected, how many of their answers are prepared, and the run that prepares
 * them. One set of calls for professors and administrators alike; the backend
 * scopes every one to the course in the path and to course staff.
 *
 * The backend decides everything that matters here (what is ready, what may be
 * started, what to say about it). This module only names the shapes and turns
 * a status into the few words and the one button the pages show.
 */

export type ApproachKey = 'base' | 'rag' | 'fineTuned' | 'fineTunedRag';

export const APPROACH_KEYS: ApproachKey[] = ['base', 'rag', 'fineTuned', 'fineTunedRag'];

/** Short labels for the per-question indicators, where the full name will not fit. */
export const APPROACH_SHORT_LABELS: Record<ApproachKey, string> = {
  base: 'Base',
  rag: 'RAG',
  fineTuned: 'FT',
  fineTunedRag: 'FT+RAG',
};

export type PreparationState =
  | 'not_started'
  | 'preparing'
  | 'stopping'
  | 'ready'
  | 'partially_ready'
  | 'stopped'
  | 'failed';

export type AnswerState = 'ready' | 'missing' | 'outdated' | 'unavailable' | 'not_expected';

export interface PreparationQuestion {
  seedId: string;
  question: string;
  origin: string;
  /** prototype | ai_generated | authored | user */
  source: string;
  sourceLabel: string;
  reviewStatus: string;
  category?: string | null;
  selected: boolean;
  eligible: boolean;
  ineligibleReason: string | null;
  /** One state per approach; null when the question is not being prepared. */
  answers: Record<ApproachKey, AnswerState> | null;
}

export interface PreparationCounts {
  selectedQuestions: number;
  expectedAnswers: number;
  readyAnswers: number;
  missingAnswers: number;
  outdatedAnswers: number;
  unavailableAnswers: number;
  fullyPreparedQuestions: number;
  preparableAnswers: number;
}

export interface PreparationCondition {
  label: string;
  expected: boolean;
  available: boolean;
  reason: string | null;
}

export interface PreparationStatus {
  courseId: string;
  /** False while classroom answers are switched off on the server. */
  enabled: boolean;
  state: PreparationState;
  running: boolean;
  stopping: boolean;
  /** "quiet" (students are using the site) or "starter" while a run holds off. */
  waiting: string | null;
  current: { seedId: string; condition: ApproachKey } | null;
  canStart: boolean;
  startBlockedReason: string | null;
  counts: PreparationCounts;
  conditions: Record<ApproachKey, PreparationCondition>;
  /** The one sentence that says where things stand, in plain words. */
  message: string;
  notices: string[];
  lastRun: {
    state: string;
    startedAt: string | null;
    finishedAt: string | null;
    reason: string | null;
    generated: number;
    failed: number;
  } | null;
  questions: PreparationQuestion[];
}

const UNREACHABLE = 'The service could not be reached.';

function path(courseId: string, suffix = ''): string {
  return `/courses/${courseId}/classroom-preparation${suffix}`;
}

export function fetchPreparationStatus(courseId: string): Promise<PreparationStatus> {
  return getJson<PreparationStatus>(
    path(courseId),
    'Classroom answers could not be loaded for this course.',
    UNREACHABLE,
  );
}

export function savePreparationSelection(
  courseId: string,
  seedIds: string[],
): Promise<PreparationStatus> {
  return sendJson<PreparationStatus>(
    'PUT',
    path(courseId, '/questions'),
    { seedIds },
    'Your question selection could not be saved.',
    UNREACHABLE,
  );
}

export function startPreparation(courseId: string): Promise<PreparationStatus> {
  return sendJson<PreparationStatus>(
    'POST',
    path(courseId, '/start'),
    undefined,
    'Preparation could not be started.',
    UNREACHABLE,
  );
}

export function stopPreparation(courseId: string): Promise<PreparationStatus> {
  return sendJson<PreparationStatus>(
    'POST',
    path(courseId, '/stop'),
    undefined,
    'Preparation could not be stopped.',
    UNREACHABLE,
  );
}

// --------------------------------------------------------------------------- //
// What the pages say about a status
// --------------------------------------------------------------------------- //

export interface PreparationPresentation {
  label: string;
  tone: StatusTone;
}

const PRESENTATION: Record<PreparationState, PreparationPresentation> = {
  not_started: { label: 'Not started', tone: 'neutral' },
  preparing: { label: 'Preparing', tone: 'progress' },
  stopping: { label: 'Stopping', tone: 'progress' },
  ready: { label: 'Ready', tone: 'success' },
  partially_ready: { label: 'Partially ready', tone: 'info' },
  stopped: { label: 'Stopped', tone: 'warning' },
  failed: { label: 'Needs attention', tone: 'danger' },
};

export function preparationPresentation(state: PreparationState): PreparationPresentation {
  return PRESENTATION[state] ?? PRESENTATION.not_started;
}

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? '' : 's'}`;
}

/** The short line beside the pill on a course page: real counts, nothing inferred. */
export function preparationSummary(status: PreparationStatus): string {
  const { selectedQuestions, fullyPreparedQuestions, readyAnswers, expectedAnswers } = status.counts;
  if (selectedQuestions === 0) {
    return 'No questions selected yet';
  }
  if (status.running) {
    return `${readyAnswers} of ${plural(expectedAnswers, 'answer')} ready`;
  }
  return `${fullyPreparedQuestions} of ${plural(selectedQuestions, 'question')} ready`;
}

export type PreparationAction =
  /** Start, or carry on with what is missing: the same request. */
  | { kind: 'start'; label: string }
  | { kind: 'stop'; label: string }
  /** A stop is in progress: shown disabled so the page never looks idle. */
  | { kind: 'stopping'; label: string }
  | null;

/**
 * The one action that makes sense right now, or none.
 *
 * Start, resume and retry are one request ("prepare whatever is missing"), so
 * they are one button whose words depend on whether anything is prepared yet.
 * When nothing can be prepared there is no button at all.
 */
export function primaryAction(status: PreparationStatus): PreparationAction {
  if (status.running) {
    return status.stopping
      ? { kind: 'stopping', label: 'Stopping…' }
      : { kind: 'stop', label: 'Stop preparation' };
  }
  if (!status.canStart) {
    return null;
  }
  return {
    kind: 'start',
    label: status.counts.readyAnswers > 0 ? 'Prepare remaining answers' : 'Prepare answers',
  };
}

/** What a per-question indicator says to a screen reader and on hover. */
export const ANSWER_STATE_LABELS: Record<AnswerState, string> = {
  ready: 'ready',
  missing: 'not prepared yet',
  outdated: 'out of date',
  unavailable: 'cannot be prepared right now',
  not_expected: 'not used by this course',
};

export const SOURCE_FILTERS: { value: string; label: string }[] = [
  { value: 'all', label: 'All sources' },
  { value: 'prototype', label: 'Staff-curated' },
  { value: 'ai_generated', label: 'AI-generated' },
  { value: 'user', label: 'Student contributions' },
  { value: 'authored', label: 'Behaviour examples' },
];

/** Behaviour examples (off-topic and refusal questions) are hidden unless asked for. */
export const BEHAVIOUR_SOURCE = 'authored';

export interface QuestionFilter {
  search: string;
  source: string;
  selectedOnly: boolean;
}

export function filterQuestions(
  questions: PreparationQuestion[],
  { search, source, selectedOnly }: QuestionFilter,
): PreparationQuestion[] {
  const needle = search.trim().toLowerCase();
  return questions.filter((question) => {
    if (selectedOnly && !question.selected) {
      return false;
    }
    if (source === 'all') {
      // A selected behaviour example stays visible so it can be deselected.
      if (question.source === BEHAVIOUR_SOURCE && !question.selected) {
        return false;
      }
    } else if (question.source !== source) {
      return false;
    }
    return !needle || question.question.toLowerCase().includes(needle);
  });
}
