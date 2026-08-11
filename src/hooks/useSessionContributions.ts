import { useCallback, useEffect, useState } from 'react';

/**
 * Ids of the questions this browser tab added, per course.
 *
 * There is no sign-in, so the application genuinely cannot know which stored
 * contributions belong to the person reading the page. The previous version
 * sidestepped that by listing everybody's — which let any student browse their
 * classmates' submitted answers.
 *
 * This is the honest middle: remember what *this tab* created, show only that,
 * and store nothing on the record itself. It is not an identity, it grants
 * nothing, it is not readable by anyone else, and it disappears with the tab.
 * A student on a second device sees an empty list, which is the correct answer
 * to "what did you add here" — not a privacy hole disguised as a history.
 */

function storageKey(courseId: string): string {
  return `sml.contributions.${courseId}`;
}

function read(courseId: string): string[] {
  try {
    const raw = window.sessionStorage.getItem(storageKey(courseId));
    if (!raw) {
      return [];
    }
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((id) => typeof id === 'string') : [];
  } catch {
    return [];
  }
}

function write(courseId: string, ids: string[]): void {
  try {
    window.sessionStorage.setItem(storageKey(courseId), JSON.stringify(ids));
  } catch {
    // Best-effort; the in-memory list still works for this page view.
  }
}

export interface SessionContributions {
  sessionIds: string[];
  rememberSessionId: (id: string) => void;
  forgetSessionId: (id: string) => void;
}

export function useSessionContributions(courseId: string): SessionContributions {
  const [sessionIds, setSessionIds] = useState<string[]>(() => read(courseId));

  useEffect(() => {
    setSessionIds(read(courseId));
  }, [courseId]);

  const rememberSessionId = useCallback(
    (id: string) => {
      setSessionIds((current) => {
        if (current.includes(id)) {
          return current;
        }
        const next = [...current, id];
        write(courseId, next);
        return next;
      });
    },
    [courseId],
  );

  const forgetSessionId = useCallback(
    (id: string) => {
      setSessionIds((current) => {
        const next = current.filter((value) => value !== id);
        write(courseId, next);
        return next;
      });
    },
    [courseId],
  );

  return { sessionIds, rememberSessionId, forgetSessionId };
}
