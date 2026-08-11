import { useCallback, useEffect, useState } from 'react';
import { subscribeToCourseModel } from '../lib/courseModelDb';
import type { CourseModelRegistry } from '../types';

export type CourseModelState =
  | { status: 'loading' }
  /** Read successfully; this course has no registered model. */
  | { status: 'none' }
  | { status: 'ready'; registry: CourseModelRegistry }
  /** Could not read. Says nothing about whether a model exists. */
  | { status: 'unavailable'; message: string };

export interface UseCourseModelResult {
  state: CourseModelState;
  retry: () => void;
}

/**
 * The registered model for one course.
 *
 * `none` and `unavailable` are kept apart on purpose: "this course has no model
 * yet" and "we could not check" lead to completely different things being said
 * to a professor, and guessing between them is how the old page ended up
 * telling CSS 360 it had no model when it has a trained one.
 */
export function useCourseModel(courseId: string | null): UseCourseModelResult {
  const [state, setState] = useState<CourseModelState>({ status: 'loading' });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!courseId) {
      setState({ status: 'none' });
      return;
    }

    setState({ status: 'loading' });

    const unsubscribe = subscribeToCourseModel(
      courseId,
      (registry) =>
        setState(registry ? { status: 'ready', registry } : { status: 'none' }),
      (message) => setState({ status: 'unavailable', message }),
    );

    return unsubscribe;
  }, [courseId, attempt]);

  const retry = useCallback(() => {
    setAttempt((current) => current + 1);
  }, []);

  return { state, retry };
}
