import * as dbApi from './dbApi';
import { pollingSubscription, type Unsubscribe } from './pollingSubscription';
import type {
  TrainingMode,
  TrainingRun,
  TrainingRunClaim,
  TrainingRunCompletion,
  TrainingRunState,
} from '../types';
import { assertValidCourseId } from './courseId';

/**
 * Training runs: the `training_runs` table, read through `/api/db`.
 *
 * The durable queue. A run is the unit of work a cluster runner claims and acts
 * on; the professor-facing `modelRequest` keeps only a `currentRunId` pointer,
 * so nothing operational — leases, attempts, job identifiers — ever has to be
 * written onto a record a professor reads.
 *
 * Course-scoped by construction: every path is built from a validated id, so
 * one course's queue can never be read or written through another's.
 *
 * The browser's responsibility ends at `queued`. It does not claim, submit, or
 * poll: it writes the record and stops. Everything after that happens on the
 * cluster, where the person running it has already authenticated normally.
 */

const RUN_STATES: readonly TrainingRunState[] = [
  'queued',
  'claimed',
  'submitted',
  'training',
  'succeeded',
  'failed',
];

/**
 * States that mean the run is finished with, one way or another.
 *
 * Everything else is outstanding work, and outstanding work is what blocks a
 * second run for the same course.
 */
const TERMINAL_RUN_STATES: readonly TrainingRunState[] = ['succeeded', 'failed'];

const MODES: readonly TrainingMode[] = ['smoke', 'full'];

export function isTrainingRunState(value: unknown): value is TrainingRunState {
  return typeof value === 'string' && RUN_STATES.includes(value as TrainingRunState);
}

export function isTerminalRunState(state: TrainingRunState): boolean {
  return TERMINAL_RUN_STATES.includes(state);
}

export function isActiveTrainingRun(run: TrainingRun): boolean {
  return !isTerminalRunState(run.state);
}

function asCount(raw: unknown): number {
  const value = Number(raw);
  return Number.isFinite(value) && value >= 0 ? Math.trunc(value) : 0;
}

function parseClaim(value: unknown): TrainingRunClaim | null {
  if (!value || typeof value !== 'object') {
    return null;
  }

  const record = value as Record<string, unknown>;

  // A claim without an owner and an expiry cannot be reasoned about: nobody
  // could say who holds it or when it may be taken again.
  if (
    typeof record.owner !== 'string' ||
    record.owner.trim() === '' ||
    typeof record.claimedAt !== 'string' ||
    typeof record.expiresAt !== 'string'
  ) {
    return null;
  }

  return {
    owner: record.owner,
    claimedAt: record.claimedAt,
    expiresAt: record.expiresAt,
  };
}

/**
 * The cluster's end-of-job report, or null.
 *
 * Only `outcome` and `receivedAt` are required, and only `outcome` is
 * constrained: everything else is assembled on a compute node from files a
 * failed run may never have written, and dropping the whole record because a
 * metric is missing would lose the one thing that was worth having — that the
 * job ended, and how.
 */
function parseCompletion(value: unknown): TrainingRunCompletion | null {
  if (!value || typeof value !== 'object') {
    return null;
  }

  const record = value as Record<string, unknown>;
  if (record.outcome !== 'succeeded' && record.outcome !== 'failed') {
    return null;
  }

  return {
    ...(record as Omit<TrainingRunCompletion, 'outcome' | 'receivedAt'>),
    outcome: record.outcome,
    receivedAt:
      typeof record.receivedAt === 'string' ? record.receivedAt : '',
  };
}

export function parseTrainingRun(runId: string, value: unknown): TrainingRun | null {
  if (!value || typeof value !== 'object') {
    return null;
  }

  const record = value as Record<string, unknown>;

  if (
    typeof record.courseId !== 'string' ||
    record.courseId.trim() === '' ||
    typeof record.enqueuedAt !== 'string' ||
    !isTrainingRunState(record.state) ||
    !MODES.includes(record.mode as TrainingMode)
  ) {
    return null;
  }

  const claim = parseClaim(record.claim);
  const completion = parseCompletion(record.completion);

  return {
    runId,
    courseId: record.courseId,
    mode: record.mode as TrainingMode,
    state: record.state,
    enqueuedAt: record.enqueuedAt,
    updatedAt:
      typeof record.updatedAt === 'string' ? record.updatedAt : record.enqueuedAt,
    datasetRef: typeof record.datasetRef === 'string' ? record.datasetRef : '',
    approvedExampleCount: asCount(record.approvedExampleCount),
    trainExamples: asCount(record.trainExamples),
    validationExamples: asCount(record.validationExamples),
    attempt: asCount(record.attempt),
    ...(typeof record.jobId === 'string' && record.jobId.trim() !== ''
      ? { jobId: record.jobId }
      : {}),
    ...(claim ? { claim } : {}),
    ...(typeof record.error === 'string' && record.error !== ''
      ? { error: record.error }
      : {}),
    ...(completion ? { completion } : {}),
  };
}

/** Reads a whole list of runs, dropping anything unreadable. */
export function parseTrainingRuns(value: unknown): TrainingRun[] {
  if (!value || typeof value !== 'object') {
    return [];
  }

  return Object.entries(value as Record<string, unknown>)
    .map(([runId, raw]) => parseTrainingRun(runId, raw))
    .filter((run): run is TrainingRun => run !== null)
    .sort((left, right) => left.enqueuedAt.localeCompare(right.enqueuedAt));
}

export function findActiveTrainingRun(runs: TrainingRun[]): TrainingRun | null {
  return runs.find(isActiveTrainingRun) ?? null;
}

/**
 * Runs for one course, read from PostgreSQL through FastAPI.
 *
 * One store: the browser reads `training_runs`, the backend writes it, and the
 * cluster runner claims from it. A run queued by `queueTraining` is therefore
 * visible to this read as soon as the enqueue returns — there is no second
 * copy that could lag behind.
 */
export async function fetchCourseTrainingRuns(courseId: string): Promise<TrainingRun[]> {
  assertValidCourseId(courseId);
  const response = await dbApi.listTrainingRuns(courseId);
  return parseTrainingRuns(
    Object.fromEntries(response.runs.map((run) => [run.runId, run])),
  );
}

/**
 * Watches a course's runs while any of them is still outstanding.
 *
 * This is the one screen where polling earns its place: a queued run is picked
 * up by a runner on another machine, so nothing in this browser knows when the
 * state changes. Polling stops as soon as every run is terminal, which is the
 * normal resting state of the page.
 */
export function subscribeToCourseTrainingRuns(
  courseId: string,
  onData: (runs: TrainingRun[]) => void,
  onError?: (message: string) => void,
): Unsubscribe {
  assertValidCourseId(courseId);

  return pollingSubscription<TrainingRun[]>({
    fetcher: () => fetchCourseTrainingRuns(courseId),
    onData,
    onError,
    shouldPoll: (runs) => findActiveTrainingRun(runs) !== null,
  });
}

export class DuplicateTrainingRunError extends Error {
  constructor() {
    super('A training run is already queued or under way for this course.');
    this.name = 'DuplicateTrainingRunError';
  }
}

/**
 * A run id that sorts by time and cannot collide in practice.
 *
 * Not an opaque key: the id is written into a record the runner echoes back in
 * logs, and a readable, sortable one is worth the four extra characters.
 * Lowercase and hyphens only, so it is a legal key and a legal path segment
 * everywhere it is used.
 */
export function generateTrainingRunId(now: Date = new Date()): string {
  const stamp = now.toISOString().replace(/[-:]/g, '').replace(/\.\d+Z$/, 'z');
  const bytes = new Uint8Array(3);
  crypto.getRandomValues(bytes);
  const suffix = Array.from(bytes, (byte) => byte.toString(36).padStart(2, '0')).join(
    '',
  );
  return `run-${stamp.toLowerCase()}-${suffix}`;
}

/*
 * Queueing lives in `queueTraining.ts`, not here.
 *
 * That module decides whether a request may be queued and points the model
 * request at the resulting run; this one only parses and reads. Callers import
 * it from there directly: re-exporting it would make this persistence module
 * depend on the orchestration one, which already depends on these parsers.
 */
