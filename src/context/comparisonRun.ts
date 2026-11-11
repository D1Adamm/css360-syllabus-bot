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
  /**
   * Which model answered, as the backend reported it. Recorded on every run
   * (not only staff testing) so a rating can say what it rated: without this,
   * the evaluations table held a winner but not the answers or the model
   * version that produced them, and a later re-activation made the research
   * data unreadable. Absent on runs stored before the field existed.
   */
  model?: string | null;
  /** The course model version that answered (fine-tuned conditions). */
  modelVersion?: string | null;
  /** Where the fine-tuned answer was generated: `vm` or `gpu`. */
  engine?: string | null;
  /**
   * True when this answer was served from the classroom answer cache instead
   * of being generated for this request. Recorded with a rating; never shown.
   * Absent when the cache is off.
   */
  cached?: boolean;
  /** The stored answer's id, so a rating can be traced to exactly that row. */
  cacheEntryId?: string | null;
}

export interface ComparisonRun {
  runId: string;
  courseId: string;
  question: string;
  /** Set when the question matched a predefined comparison. */
  matchedComparisonId: string | null;
  createdAt: string;
  responses: Record<ModelKey, ComparisonRunResponse>;
  /**
   * Set once this run has been rated, so Evaluate shows the thanks rather than
   * the form again and Compare stops offering "rate your last comparison".
   * The backend refuses a second rating of the same run in any case; this is
   * what keeps the page from asking for one.
   */
  ratedAt?: string;
  /** The stored rating's id, when the backend returned one. */
  evaluationId?: string | null;
}

/** The snapshot an evaluation stores of one approach's answer. */
export function snapshotOf(response: ComparisonRunResponse): ComparisonRunResponse {
  return {
    text: response.text,
    error: response.error ?? null,
    sources: response.sources ?? [],
    model: response.model ?? null,
    modelVersion: response.modelVersion ?? null,
    engine: response.engine ?? null,
    // Only when the answer has a stored row; otherwise the snapshot keeps the
    // shape it has always had.
    ...(response.cacheEntryId
      ? { cacheEntryId: response.cacheEntryId, cached: response.cached === true }
      : {}),
  };
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
