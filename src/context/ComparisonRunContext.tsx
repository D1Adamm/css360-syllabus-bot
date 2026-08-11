import { createContext, useCallback, useContext, useMemo, useState } from 'react';
import {
  readStoredRun,
  removeStoredRun,
  writeStoredRun,
  type ComparisonRun,
} from './comparisonRun';

export type { ComparisonRun, ComparisonRunResponse } from './comparisonRun';

interface ComparisonRunContextValue {
  getRun: (courseId: string) => ComparisonRun | null;
  saveRun: (run: ComparisonRun) => void;
  clearRun: (courseId: string) => void;
}

const ComparisonRunContext = createContext<ComparisonRunContextValue | null>(null);

export function ComparisonRunProvider({ children }: { children: React.ReactNode }) {
  // Kept in state as well as storage so navigating between Compare and
  // Evaluate re-renders without a storage read race.
  const [runs, setRuns] = useState<Record<string, ComparisonRun>>({});

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

  const value = useMemo(
    () => ({ getRun, saveRun, clearRun }),
    [getRun, saveRun, clearRun],
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
