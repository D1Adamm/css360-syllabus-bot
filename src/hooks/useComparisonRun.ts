import { useCallback } from 'react';
import { useComparisonRunStore } from '../context/ComparisonRunContext';
import { IDLE_STATES, type ApproachStates } from '../context/comparisonRunner';

export type { ApproachState, ApproachStates } from '../context/comparisonRunner';

export interface UseComparisonRunResult {
  states: ApproachStates;
  activeQuestion: string;
  isRunning: boolean;
  run: (question: string, matchedComparisonId: string | null) => void;
}

/**
 * The Compare page's view of this course's comparison.
 *
 * The comparison itself lives in `ComparisonRunProvider`, above the routes, so
 * this hook holds no state of its own: a page that unmounts mid-comparison and
 * mounts again reads the same question, the same pending cards and whichever
 * answers have arrived since. Asking again while one is running does nothing.
 */
export function useComparisonRun(courseId: string): UseComparisonRunResult {
  const { getActive, startComparison } = useComparisonRunStore();
  const active = getActive(courseId);

  const run = useCallback(
    (question: string, matchedComparisonId: string | null) =>
      startComparison(courseId, question, matchedComparisonId),
    [courseId, startComparison],
  );

  return {
    states: active?.states ?? IDLE_STATES,
    activeQuestion: active?.question ?? '',
    isRunning: active?.isRunning ?? false,
    run,
  };
}
