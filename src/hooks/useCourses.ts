import { useCallback, useEffect, useState } from 'react';
import { subscribeToCourses, type CourseListItem } from '../lib/coursesDb';

export type CoursesState =
  | { status: 'loading' }
  | { status: 'ready'; courses: CourseListItem[] }
  | { status: 'error'; message: string };

export interface UseCoursesResult {
  state: CoursesState;
  /** Re-subscribes. Safe to call from a retry button. */
  retry: () => void;
}

/**
 * Live list of courses.
 *
 * Wraps the same `subscribeToCourses` call the course picker has always used,
 * so ordering and parsing behaviour are unchanged; only the callers differ.
 *
 * `retry` tears down and re-establishes the subscription. A realtime listener
 * that failed once will not recover on its own, so a dead-end error banner is
 * genuinely dead — this gives it a way back.
 */
export function useCourses(): UseCoursesResult {
  const [state, setState] = useState<CoursesState>({ status: 'loading' });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    setState({ status: 'loading' });

    const unsubscribe = subscribeToCourses(
      (courses) => setState({ status: 'ready', courses }),
      (message) => setState({ status: 'error', message }),
    );

    return unsubscribe;
  }, [attempt]);

  const retry = useCallback(() => {
    setAttempt((current) => current + 1);
  }, []);

  return { state, retry };
}
