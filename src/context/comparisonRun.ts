import type { ModelKey } from '../types';

/**
 * The comparison run that Compare hands to Evaluate.
 *
 * Before this existed, Evaluate rated four canned answers from
 * `comparisonData.json` while Compare generated four live ones — so a student
 * clicking "Evaluate these responses" rated text they had never seen. This is
 * the record that fixes that.
 *
 * Mirrored into `sessionStorage` so a refresh, or opening Evaluate directly,
 * still finds the run. It is per-tab and short-lived on purpose: it is working
 * state, not a result. The rating itself is what gets persisted.
 *
 * Kept separate from the provider so these helpers and types can be imported
 * without pulling in a React component.
 */

export interface ComparisonRunResponse {
  text: string;
  /** Student-facing message when this approach could not answer. */
  error: string | null;
  sources: string[];
}

export interface ComparisonRun {
  runId: string;
  courseId: string;
  question: string;
  /** Set when the question matched a predefined comparison. */
  matchedComparisonId: string | null;
  createdAt: string;
  responses: Record<ModelKey, ComparisonRunResponse>;
}

const MODEL_KEYS: ModelKey[] = ['base', 'rag', 'fineTuned', 'fineTunedRag'];

function storageKey(courseId: string): string {
  return `sml.run.${courseId}`;
}

function isComparisonRun(value: unknown): value is ComparisonRun {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const run = value as ComparisonRun;
  if (
    typeof run.runId !== 'string' ||
    typeof run.courseId !== 'string' ||
    typeof run.question !== 'string' ||
    typeof run.responses !== 'object' ||
    run.responses === null
  ) {
    return false;
  }
  // Every approach must be present, so Evaluate never renders a partial grid.
  return MODEL_KEYS.every((key) => {
    const response = run.responses[key];
    return (
      typeof response === 'object' &&
      response !== null &&
      typeof response.text === 'string'
    );
  });
}

export function readStoredRun(courseId: string): ComparisonRun | null {
  try {
    const raw = window.sessionStorage.getItem(storageKey(courseId));
    if (!raw) {
      return null;
    }
    const parsed: unknown = JSON.parse(raw);
    return isComparisonRun(parsed) ? parsed : null;
  } catch {
    // Unusable storage or malformed JSON simply means "no run yet".
    return null;
  }
}

export function writeStoredRun(run: ComparisonRun): void {
  try {
    window.sessionStorage.setItem(storageKey(run.courseId), JSON.stringify(run));
  } catch {
    // Persisting is best-effort; the in-memory copy still works this session.
  }
}

export function removeStoredRun(courseId: string): void {
  try {
    window.sessionStorage.removeItem(storageKey(courseId));
  } catch {
    // Nothing to do.
  }
}
