/**
 * The replacement for the realtime listeners this app used before the cutover.
 *
 * `onValue` gave every screen a live subscription for free. Almost none of them
 * needed one: a course's title, its seed list, its registered model, and its
 * evaluations only change when someone in this application changes them, and
 * the screen that changed them already knows. Rebuilding push semantics over
 * HTTP — websockets, SSE, a subscription registry — would be a large amount of
 * machinery to keep static data live.
 *
 * So this keeps the shape callers already use, `subscribe(onData, onError)`
 * returning an unsubscribe, and fetches once. Polling is opt-in per call site
 * and conditional: `shouldPoll` decides, from the value just fetched, whether
 * another read is worth making. Starter generation returns true while a job is
 * queued or running and false the moment it reaches a terminal state, so the
 * timer exists only while something is genuinely moving and stops on its own
 * without anyone having to remember to cancel it.
 *
 * On error it stops. A failing endpoint polled every few seconds produces a
 * stream of identical failures and keeps the tab busy; every caller already
 * exposes a retry, which is the honest way back.
 */

export type Unsubscribe = () => void;

export const DEFAULT_POLL_INTERVAL_MS = 5000;

export interface PollingSubscriptionOptions<T> {
  /** One read. Rejections are reported through `onError`. */
  fetcher: () => Promise<T>;
  onData: (value: T) => void;
  onError?: (message: string) => void;
  /**
   * Whether to schedule another read after this value. Omit for data that does
   * not change on its own — the common case, and it means one fetch only.
   */
  shouldPoll?: (value: T) => boolean;
  intervalMs?: number;
}

function messageFor(error: unknown): string {
  return error instanceof Error && error.message.trim() !== ''
    ? error.message
    : 'Something went wrong while loading this. Try again in a moment.';
}

export function pollingSubscription<T>({
  fetcher,
  onData,
  onError,
  shouldPoll,
  intervalMs = DEFAULT_POLL_INTERVAL_MS,
}: PollingSubscriptionOptions<T>): Unsubscribe {
  let cancelled = false;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const clear = () => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  };

  const run = async () => {
    let value: T;

    try {
      value = await fetcher();
    } catch (error) {
      // A response that arrives after unmount, or after the course changed,
      // belongs to a subscription nobody is listening to any more. Delivering
      // it would repopulate a screen that has already moved on — which is
      // exactly how one course's data would appear under another's heading.
      if (cancelled) {
        return;
      }
      onError?.(messageFor(error));
      return;
    }

    if (cancelled) {
      return;
    }

    onData(value);

    if (shouldPoll?.(value)) {
      timer = setTimeout(() => {
        void run();
      }, intervalMs);
    }
  };

  void run();

  return () => {
    cancelled = true;
    clear();
  };
}
