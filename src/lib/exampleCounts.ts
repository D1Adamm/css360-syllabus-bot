import type { CourseSeedReviewRecord, SeedReviewStatus } from './api';

/**
 * Review-status arithmetic for course examples.
 *
 * These counts come from the examples themselves. They are deliberately not
 * derived from the approved-export status: an export is a training artefact
 * that may be stale, missing, or newer than the current review state, and
 * "48 approved examples" must mean exactly that.
 */

export type ExampleStatus = SeedReviewStatus | string;

export interface ExampleCounts {
  total: number;
  /** Waiting for a decision. */
  pending: number;
  approved: number;
  rejected: number;
  edited: number;
}

export const EMPTY_COUNTS: ExampleCounts = {
  total: 0,
  pending: 0,
  approved: 0,
  rejected: 0,
  edited: 0,
};

/** Older records store the status on `status`; newer ones on `reviewStatus`. */
export function resolveExampleStatus(seed: CourseSeedReviewRecord): ExampleStatus {
  return String(seed.reviewStatus || seed.status || 'generated')
    .trim()
    .toLowerCase();
}

export function exampleWasEdited(seed: CourseSeedReviewRecord): boolean {
  if (seed.wasEdited === true) {
    return true;
  }
  if (resolveExampleStatus(seed) === 'edited') {
    return true;
  }
  if (String(seed.originalQuestion || '').trim()) {
    return true;
  }
  if (String(seed.originalAnswer || '').trim()) {
    return true;
  }
  return false;
}

export function exampleQuestion(seed: CourseSeedReviewRecord): string {
  return String(seed.question || seed.instruction || '').trim();
}

export function exampleAnswer(seed: CourseSeedReviewRecord): string {
  return String(seed.answer || seed.response || '').trim();
}

export function countExamples(seeds: CourseSeedReviewRecord[]): ExampleCounts {
  const counts: ExampleCounts = { ...EMPTY_COUNTS, total: seeds.length };

  for (const seed of seeds) {
    switch (resolveExampleStatus(seed)) {
      case 'approved':
        counts.approved += 1;
        break;
      case 'rejected':
        counts.rejected += 1;
        break;
      case 'edited':
        counts.edited += 1;
        break;
      default:
        // Anything not yet decided counts as pending, including records with
        // an unrecognised status from an older run.
        counts.pending += 1;
    }
  }

  return counts;
}
