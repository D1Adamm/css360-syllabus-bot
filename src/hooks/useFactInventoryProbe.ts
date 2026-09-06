import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError, requestFactInventory, type FactInventoryResponse } from '../lib/adminApi';

/**
 * Poll every this often while the backend reports the build is running.
 * Extraction takes minutes at best, so anything faster is noise.
 */
export const FACT_INVENTORY_POLL_INTERVAL_MS = 5_000;

/**
 * Stop polling after this many polls and hand control back to the reader.
 *
 * Two minutes. A rebuild of a long syllabus takes far longer than that on the
 * CPU, and a control that spins for an hour is indistinguishable from one that
 * is broken — which is exactly what the page used to do. After the budget the
 * page says the build is still running on the server and offers to check
 * again; the server keeps working either way.
 */
export const FACT_INVENTORY_MAX_POLLS = 24;

export type FactInventoryProbeState =
  | { status: 'idle' }
  /** A request is out, or the build is running and the page is polling. */
  | { status: 'building'; startedAt: string | null; polls: number }
  /** The build is still running on the server; polling has paused. */
  | { status: 'pending'; startedAt: string | null }
  | { status: 'ok'; data: FactInventoryResponse }
  | { status: 'failed'; message: string };

export interface FactInventoryProbe {
  state: FactInventoryProbeState;
  /** Start, or check again. Safe to call in any state. */
  inspect: () => void;
}

function errorText(error: unknown): string {
  return error instanceof ApiError ? error.message : String(error);
}

/**
 * The fact-inventory diagnostic, as a state machine that always terminates.
 *
 * Every path out of `building` is bounded: a result, an error, or the poll
 * budget running out. Timers are cancelled on unmount and superseded by a
 * later `inspect`, so a response from an earlier click can never overwrite a
 * later one and a page that was navigated away from does nothing further.
 */
export function useFactInventoryProbe(courseId: string): FactInventoryProbe {
  const [state, setState] = useState<FactInventoryProbeState>({ status: 'idle' });
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const runRef = useRef(0);
  const mountedRef = useRef(true);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      clearTimer();
      // Invalidate any request still in flight.
      runRef.current += 1;
    };
  }, [clearTimer]);

  const check = useCallback(
    async (run: number, polls: number, startedAt: string | null) => {
      let result: Awaited<ReturnType<typeof requestFactInventory>>;
      try {
        result = await requestFactInventory(courseId);
      } catch (error) {
        if (!mountedRef.current || run !== runRef.current) {
          return;
        }
        setState({ status: 'failed', message: errorText(error) });
        return;
      }

      if (!mountedRef.current || run !== runRef.current) {
        return;
      }

      if (result.status === 'ready') {
        setState({ status: 'ok', data: result.inventory });
        return;
      }

      const nextPolls = polls + 1;
      const since = result.startedAt ?? startedAt;
      if (nextPolls >= FACT_INVENTORY_MAX_POLLS) {
        setState({ status: 'pending', startedAt: since });
        return;
      }

      setState({ status: 'building', startedAt: since, polls: nextPolls });
      timerRef.current = setTimeout(() => {
        timerRef.current = null;
        void check(run, nextPolls, since);
      }, FACT_INVENTORY_POLL_INTERVAL_MS);
    },
    [courseId],
  );

  const inspect = useCallback(() => {
    clearTimer();
    runRef.current += 1;
    const run = runRef.current;
    setState({ status: 'building', startedAt: null, polls: 0 });
    void check(run, 0, null);
  }, [check, clearTimer]);

  return { state, inspect };
}
