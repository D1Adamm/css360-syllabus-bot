import { useCallback, useEffect, useRef, useState } from 'react';
import {
  IDLE_STATES,
  LOADING_STATES,
  runFourConditions,
  type ApproachStates,
} from '../context/comparisonRunner';
import type { Role } from '../context/role';

export interface UseStaffComparisonResult {
  states: ApproachStates;
  question: string;
  isRunning: boolean;
  ask: (question: string) => void;
}

/**
 * A staff member's own comparison, for the Professor "Ask questions" tab.
 *
 * The same four requests the student Compare page makes (`runFourConditions`),
 * and deliberately none of its bookkeeping. The student flow's comparisons
 * live in `ComparisonRunProvider` and are mirrored to `sessionStorage` for
 * Evaluate; a staff question kept there could turn up on a course's Evaluate
 * page, where anyone holding a participant session would rate it into the
 * study. Here the comparison is this component's state and nothing else:
 * nothing is stored, and leaving the tab discards it.
 *
 * `audience` is the signed-in staff role, so a failed condition reads in
 * staff terms (an administrator sees the backend's own reason). Each answer
 * reports the model and version that produced it.
 */
export function useStaffComparison(courseId: string, audience: Role): UseStaffComparisonResult {
  const [states, setStates] = useState<ApproachStates>(IDLE_STATES);
  const [question, setQuestion] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  // Synchronous: a double click must not slip a second set of requests past
  // a not-yet-rendered flag.
  const running = useRef(false);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const ask = useCallback(
    (raw: string) => {
      const trimmed = raw.trim();
      if (!trimmed || running.current) {
        return;
      }
      running.current = true;
      setQuestion(trimmed);
      setStates(LOADING_STATES);
      setIsRunning(true);

      void runFourConditions(
        courseId,
        trimmed,
        {
          onSettled: (key, state) => {
            if (mounted.current) {
              setStates((current) => ({ ...current, [key]: state }));
            }
          },
        },
        { audience, includeModel: true },
      ).finally(() => {
        running.current = false;
        if (mounted.current) {
          setIsRunning(false);
        }
      });
    },
    [courseId, audience],
  );

  return { states, question, isRunning, ask };
}
