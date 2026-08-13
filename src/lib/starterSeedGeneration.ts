import type { CourseMetadata, StoredStarterSeedGeneration } from '../types';

/**
 * Whether a course's starter examples are still being written.
 *
 * The state is already durable and already course-scoped: the generation job
 * writes it to `courses/{courseId}/metadata/starterSeedGeneration` as it goes.
 * Nothing here starts, polls, or infers anything — it reads the record the job
 * left and says what it means to a professor. A refresh therefore shows the
 * same thing, because the answer never lived in the browser.
 *
 * Four states, and no percentage. The job reports counts against a target, not
 * progress: an example is produced, validated, and either saved or dropped, so
 * a bar would be a number we made up. "This may take several minutes" is the
 * true statement, and it is the one a professor can act on.
 */

export type StarterGenerationState =
  /** No generation has ever been recorded for this course. */
  | 'not_started'
  /** Queued or running. A professor is waiting. */
  | 'generating'
  /** Finished with examples to review. */
  | 'ready'
  /** Finished without producing anything usable. */
  | 'failed';

export interface StarterGeneration {
  state: StarterGenerationState;
  /** Examples actually saved, when the job recorded a figure. */
  generatedCount: number;
  startedAt: string | null;
  completedAt: string | null;
}

const NOT_STARTED: StarterGeneration = {
  state: 'not_started',
  generatedCount: 0,
  startedAt: null,
  completedAt: null,
};

/**
 * The job's vocabulary, mapped to the four states a professor is shown.
 *
 * `queued` and `generating` are one wait as far as anyone waiting is concerned;
 * splitting them would show a professor a distinction between "about to start"
 * and "started" that changes nothing they would do.
 *
 * `partial` maps to `ready`, deliberately. It means the job produced fewer
 * examples than it aimed for — but it produced some, and a queue of real
 * examples to review is not a failure. Calling it one would send a professor
 * looking for a problem in a course that is working.
 */
function mapStatus(status: unknown): StarterGenerationState | null {
  switch (status) {
    case 'queued':
    case 'generating':
      return 'generating';
    case 'ready':
    case 'partial':
      return 'ready';
    case 'failed':
      return 'failed';
    default:
      // Includes `not_started`, an absent status, and anything a future job
      // version writes that this build has never heard of. Claiming failure
      // for an unknown word would be worse than saying nothing.
      return null;
  }
}

function asCount(raw: unknown): number {
  const value = Number(raw);
  return Number.isFinite(value) && value >= 0 ? Math.trunc(value) : 0;
}

function asTimestamp(raw: unknown): string | null {
  return typeof raw === 'string' && raw.trim() !== '' ? raw : null;
}

export function parseStarterGeneration(value: unknown): StarterGeneration {
  if (!value || typeof value !== 'object') {
    return NOT_STARTED;
  }

  const record = value as StoredStarterSeedGeneration;
  const state = mapStatus(record.status);

  if (state === null) {
    return NOT_STARTED;
  }

  return {
    state,
    // What was saved, not what was attempted: only saved examples are ones a
    // professor can actually open.
    generatedCount: asCount(record.savedCount ?? record.finalCount),
    startedAt: asTimestamp(record.startedAt),
    completedAt: asTimestamp(record.completedAt),
  };
}

/**
 * Reads one course's generation state from its metadata.
 *
 * Course-scoped by construction: the record is a field of that course's own
 * metadata, so a page rendering course A can only ever read A's state.
 */
export function getStarterGeneration(
  metadata: CourseMetadata | null,
): StarterGeneration {
  return metadata ? parseStarterGeneration(metadata.starterSeedGeneration) : NOT_STARTED;
}

export function isGeneratingStarterExamples(metadata: CourseMetadata | null): boolean {
  return getStarterGeneration(metadata).state === 'generating';
}

export interface StarterGenerationPresentation {
  title: string;
  detail: string;
}

/**
 * Professor-facing wording for a generation state.
 *
 * Nothing from the stored record appears here beyond the state itself. The job
 * records the model's or the backend's own error text, and that text names
 * services, hosts and limits a professor has no way to act on — so the failure
 * wording is written here, in full, and the stored reason is left for whoever
 * operates the system.
 */
export function describeStarterGeneration(
  state: StarterGenerationState,
): StarterGenerationPresentation | null {
  switch (state) {
    case 'generating':
      return {
        title: 'Generating starter examples…',
        detail:
          "We're creating example questions from your syllabus. This may take " +
          'several minutes. You can leave this page and come back.',
      };
    case 'failed':
      return {
        title: "We couldn't create starter examples",
        detail:
          'Your syllabus is saved and nothing you have done is lost. The ' +
          'project administrator has been notified and will follow up.',
      };
    default:
      // `ready` and `not_started` need no explanation of their own: the review
      // queue and its existing empty state already say everything true.
      return null;
  }
}
