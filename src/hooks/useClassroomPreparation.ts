import { useCallback, useEffect, useRef, useState } from 'react';
import {
  fetchPreparationStatus,
  savePreparationSelection,
  startPreparation,
  stopPreparation,
  type PreparationStatus,
} from '../lib/classroomPreparation';
import { ApiError } from '../lib/httpClient';

export type ClassroomPreparationState =
  | { status: 'loading' }
  | { status: 'ready'; preparation: PreparationStatus }
  /** Could not read. Says nothing about what is or is not prepared. */
  | { status: 'unavailable'; message: string };

export interface UseClassroomPreparationResult {
  state: ClassroomPreparationState;
  /** True while a start, stop or selection save is on its way. */
  busy: boolean;
  /** The last action's failure, in plain words; cleared by the next action. */
  actionError: string | null;
  retry: () => void;
  start: () => void;
  stop: () => void;
  /** Make the course's selection exactly these questions. */
  saveSelection: (seedIds: string[]) => void;
}

const GENERIC_FAILURE = 'Classroom answers could not be reached just now. Please try again in a moment.';

/**
 * What to tell staff when a request fails.
 *
 * A refusal (409) or a rejected selection (422) is the backend explaining
 * itself in a sentence written for this page, so it is shown as it is.
 * Anything else is an outage, which gets one calm sentence and no detail.
 */
export function describePreparationError(error: unknown): string {
  if (error instanceof ApiError && (error.status === 409 || error.status === 422) && error.message) {
    return error.message;
  }
  return GENERIC_FAILURE;
}

/** How often the page asks again while a run is in progress. */
export const RUNNING_POLL_MS = 2500;

/**
 * One course's classroom preparation, read from the server.
 *
 * Nothing about a run lives here: the run belongs to the backend, so leaving
 * the page, refreshing it or opening it elsewhere shows the same thing. The
 * hook re-reads the status every few seconds only while a run is in progress
 * and stops asking when it is not.
 *
 * Selection saves are "latest wins": each save replaces the whole selection,
 * so when checkboxes are clicked faster than the server answers, only the
 * newest selection is sent next and an older answer never overwrites a newer
 * choice on screen.
 */
export function useClassroomPreparation(courseId: string | null): UseClassroomPreparationResult {
  const [state, setState] = useState<ClassroomPreparationState>({ status: 'loading' });
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  // Guards against a response for a course, or a request, that is no longer current.
  const generation = useRef(0);
  const saving = useRef(false);
  const queuedSelection = useRef<string[] | null>(null);

  useEffect(() => {
    generation.current += 1;
    const current = generation.current;
    queuedSelection.current = null;
    setActionError(null);
    if (!courseId) {
      setState({ status: 'unavailable', message: 'No course selected.' });
      return;
    }
    setState({ status: 'loading' });
    fetchPreparationStatus(courseId)
      .then((preparation) => {
        if (generation.current === current) {
          setState({ status: 'ready', preparation });
        }
      })
      .catch((error: unknown) => {
        if (generation.current === current) {
          setState({ status: 'unavailable', message: describePreparationError(error) });
        }
      });
  }, [courseId, attempt]);

  const running = state.status === 'ready' && state.preparation.running;

  useEffect(() => {
    if (!courseId || !running) {
      return undefined;
    }
    const current = generation.current;
    const timer = window.setInterval(() => {
      fetchPreparationStatus(courseId)
        .then((preparation) => {
          // A save in flight will bring its own, newer status.
          if (generation.current === current && !saving.current) {
            setState({ status: 'ready', preparation });
          }
        })
        .catch(() => {
          // A missed poll is not worth a message; the next one will say.
        });
    }, RUNNING_POLL_MS);
    return () => window.clearInterval(timer);
  }, [courseId, running]);

  const act = useCallback(
    (request: (id: string) => Promise<PreparationStatus>) => {
      if (!courseId) {
        return;
      }
      const current = generation.current;
      setBusy(true);
      setActionError(null);
      request(courseId)
        .then((preparation) => {
          if (generation.current === current) {
            setState({ status: 'ready', preparation });
          }
        })
        .catch((error: unknown) => {
          if (generation.current === current) {
            setActionError(describePreparationError(error));
            // The refusal may be about a state this page had not seen yet.
            fetchPreparationStatus(courseId)
              .then((preparation) => {
                if (generation.current === current) {
                  setState({ status: 'ready', preparation });
                }
              })
              .catch(() => undefined);
          }
        })
        .finally(() => {
          if (generation.current === current) {
            setBusy(false);
          }
        });
    },
    [courseId],
  );

  const start = useCallback(() => act(startPreparation), [act]);
  const stop = useCallback(() => act(stopPreparation), [act]);

  const sendSelection = useCallback(
    (seedIds: string[]) => {
      if (!courseId) {
        return;
      }
      const current = generation.current;
      saving.current = true;
      setBusy(true);
      setActionError(null);
      savePreparationSelection(courseId, seedIds)
        .then((preparation) => {
          // Shown only when no newer choice is waiting to be sent.
          if (generation.current === current && queuedSelection.current === null) {
            setState({ status: 'ready', preparation });
          }
        })
        .catch((error: unknown) => {
          if (generation.current !== current) {
            return;
          }
          queuedSelection.current = null;
          setActionError(describePreparationError(error));
          // Put the checkboxes back to what the server actually holds.
          fetchPreparationStatus(courseId)
            .then((preparation) => {
              if (generation.current === current) {
                setState({ status: 'ready', preparation });
              }
            })
            .catch(() => undefined);
        })
        .finally(() => {
          if (generation.current !== current) {
            return;
          }
          const next = queuedSelection.current;
          queuedSelection.current = null;
          if (next !== null) {
            sendSelection(next);
          } else {
            saving.current = false;
            setBusy(false);
          }
        });
    },
    [courseId],
  );

  const saveSelection = useCallback(
    (seedIds: string[]) => {
      // Show the choice at once; the server's answer confirms or corrects it.
      const chosen = new Set(seedIds);
      setState((previous) =>
        previous.status === 'ready'
          ? {
              status: 'ready',
              preparation: {
                ...previous.preparation,
                questions: previous.preparation.questions.map((question) => ({
                  ...question,
                  selected: chosen.has(question.seedId),
                })),
              },
            }
          : previous,
      );
      if (saving.current) {
        queuedSelection.current = seedIds;
      } else {
        sendSelection(seedIds);
      }
    },
    [sendSelection],
  );

  const retry = useCallback(() => setAttempt((value) => value + 1), []);

  return { state, busy, actionError, retry, start, stop, saveSelection };
}
