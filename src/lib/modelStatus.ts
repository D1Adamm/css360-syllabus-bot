/**
 * Course model state.
 *
 * THERE IS NO BACKEND FOR THIS YET. Everything a professor would want to know —
 * has a model been requested, is it training, is it ready for this course — has
 * no endpoint and no stored record. This module exists so that fact lives in
 * exactly one file instead of being smeared across the professor pages.
 *
 * Why the one related endpoint cannot answer it: `GET /fine-tuned/health`
 * describes a single shared inference service. It reports whether *some*
 * adapter is loaded, not which course it belongs to. Reading it as "this
 * course's model is ready" would be wrong for every course but one, so we do
 * not read it here at all.
 *
 * See `docs/frontend-backend-gaps.md` for the endpoints this needs.
 */

export type CourseModelState =
  /** No request has been made. The only state we can currently be sure of. */
  | 'not_requested'
  /** A request exists and is queued. Needs a request record to be real. */
  | 'requested'
  /** Training is running. Needs job status to be real. */
  | 'preparing'
  /** A model for this course is deployed. Needs a per-course model registry. */
  | 'ready'
  /** Training failed. Needs job status to be real. */
  | 'needs_attention'
  /** We genuinely cannot tell. This is what the UI shows today. */
  | 'unknown';

export interface CourseModelStatus {
  state: CourseModelState;
  /** True while no backend can answer the question. */
  isPlaceholder: boolean;
}

/**
 * What we can honestly report about a course's model right now.
 *
 * Always `unknown`. When the backend gains a request record and job status,
 * this becomes a real read and every caller updates for free.
 */
export function getCourseModelStatus(): CourseModelStatus {
  return { state: 'unknown', isPlaceholder: true };
}

/** Minimum approved examples before requesting a model is worthwhile. */
export const RECOMMENDED_APPROVED_EXAMPLES = 30;

export interface ModelReadiness {
  approved: number;
  /** Enough approved examples to be worth training on. */
  hasEnough: boolean;
  remaining: number;
}

/**
 * Whether a course has enough approved examples to be worth training on.
 *
 * This *is* answerable today, because it depends only on review status. It is
 * deliberately separate from `getCourseModelStatus` so that a real readiness
 * figure is never mistaken for a real model state.
 */
export function getModelReadiness(approved: number): ModelReadiness {
  return {
    approved,
    hasEnough: approved >= RECOMMENDED_APPROVED_EXAMPLES,
    remaining: Math.max(0, RECOMMENDED_APPROVED_EXAMPLES - approved),
  };
}
