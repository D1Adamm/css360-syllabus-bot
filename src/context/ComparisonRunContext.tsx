import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';
import {
  readStoredRun,
  removeStoredRun,
  writeStoredRun,
  type ComparisonRun,
} from './comparisonRun';
import {
  generateRunId,
  LOADING_STATES,
  runFourConditions,
  type ApproachStates,
} from './comparisonRunner';

export type { ComparisonRun, ComparisonRunResponse } from './comparisonRun';

/** The comparison a course is asking right now, or last asked this visit. */
export interface ActiveComparison {
  question: string;
  states: ApproachStates;
  isRunning: boolean;
}

interface ComparisonRunContextValue {
  getRun: (courseId: string) => ComparisonRun | null;
  saveRun: (run: ComparisonRun) => void;
  clearRun: (courseId: string) => void;
  /** The live comparison for a course, if one was asked this visit. */
  getActive: (courseId: string) => ActiveComparison | null;
  /**
   * Ask the four conditions. Ignored while this course already has a
   * comparison running, so a remounted page or a double click never sends a
   * second set of requests.
   */
  startComparison: (courseId: string, question: string, matchedComparisonId: string | null) => void;
}

const ComparisonRunContext = createContext<ComparisonRunContextValue | null>(null);

/**
 * Comparison state for the whole application, per course.
 *
 * Mounted above the routes, so it outlives any page: a student can ask, go to
 * Contribute or Home while the answers generate, and come back to the same
 * question with whichever answers have arrived. The requests belong to this
 * provider, not to the Compare page, so leaving the page cancels nothing.
 *
 * Completed runs are also mirrored to `sessionStorage` for Evaluate, as before.
 * A comparison still running when the tab is reloaded is not recovered: the
 * student asks again.
 */
export function ComparisonRunProvider({ children }: { children: React.ReactNode }) {
  // Kept in state as well as storage so navigating between Compare and
  // Evaluate re-renders without a storage read race.
  const [runs, setRuns] = useState<Record<string, ComparisonRun>>({});
  const [active, setActive] = useState<Record<string, ActiveComparison>>({});
  // Synchronous per-course lock: state updates are asynchronous, and a double
  // click must not slip a second comparison past a not-yet-rendered flag.
  const running = useRef<Set<string>>(new Set());

  const getRun = useCallback(
    (courseId: string) => runs[courseId] ?? readStoredRun(courseId),
    [runs],
  );

  const saveRun = useCallback((run: ComparisonRun) => {
    writeStoredRun(run);
    setRuns((current) => ({ ...current, [run.courseId]: run }));
  }, []);

  const clearRun = useCallback((courseId: string) => {
    removeStoredRun(courseId);
    setRuns((current) => {
      const next = { ...current };
      delete next[courseId];
      return next;
    });
  }, []);

  const getActive = useCallback((courseId: string) => active[courseId] ?? null, [active]);

  const startComparison = useCallback(
    (courseId: string, question: string, matchedComparisonId: string | null) => {
      const trimmed = question.trim();
      if (!trimmed || running.current.has(courseId)) {
        return;
      }
      running.current.add(courseId);
      setActive((current) => ({
        ...current,
        [courseId]: { question: trimmed, states: LOADING_STATES, isRunning: true },
      }));

      void runFourConditions(courseId, trimmed, {
        onSettled: (key, state) => {
          setActive((current) => {
            const entry = current[courseId];
            if (!entry) {
              return current;
            }
            return {
              ...current,
              [courseId]: { ...entry, states: { ...entry.states, [key]: state } },
            };
          });
        },
      }).then((responses) => {
        running.current.delete(courseId);
        setActive((current) => {
          const entry = current[courseId];
          return entry ? { ...current, [courseId]: { ...entry, isRunning: false } } : current;
        });
        saveRun({
          runId: generateRunId(),
          courseId,
          question: trimmed,
          matchedComparisonId,
          createdAt: new Date().toISOString(),
          responses,
        });
      });
    },
    [saveRun],
  );

  const value = useMemo(
    () => ({ getRun, saveRun, clearRun, getActive, startComparison }),
    [getRun, saveRun, clearRun, getActive, startComparison],
  );

  return (
    <ComparisonRunContext.Provider value={value}>{children}</ComparisonRunContext.Provider>
  );
}

export function useComparisonRunStore(): ComparisonRunContextValue {
  const value = useContext(ComparisonRunContext);
  if (!value) {
    throw new Error('useComparisonRunStore requires a ComparisonRunProvider.');
  }
  return value;
}
