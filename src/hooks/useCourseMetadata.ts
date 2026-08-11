import { useCallback, useEffect, useState } from 'react';
import { subscribeToCourseMetadata } from '../lib/coursesDb';
import type { CourseMetadata } from '../types';

export type CourseMetadataState =
  | { status: 'loading' }
  | { status: 'ready'; metadata: CourseMetadata }
  /** The record was read and there is genuinely no course there. */
  | { status: 'missing' }
  /** We could not read it. Says nothing about whether it exists. */
  | { status: 'unavailable'; message: string };

export interface UseCourseMetadataResult {
  state: CourseMetadataState;
  /** Convenience for chrome that only wants the values when present. */
  metadata: CourseMetadata | null;
  retry: () => void;
}

/**
 * Live metadata for one course.
 *
 * The distinction between `missing` and `unavailable` is the point. Collapsing
 * both to `null` meant a temporary read failure rendered as "No syllabus added
 * yet" — telling a professor their upload had vanished when in fact we simply
 * could not check. Callers that show status must handle the two separately.
 */
export function useCourseMetadata(courseId: string | null): UseCourseMetadataResult {
  const [state, setState] = useState<CourseMetadataState>({ status: 'loading' });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!courseId) {
      setState({ status: 'missing' });
      return;
    }

    setState({ status: 'loading' });

    const unsubscribe = subscribeToCourseMetadata(
      courseId,
      (next) =>
        setState(next ? { status: 'ready', metadata: next } : { status: 'missing' }),
      (message) => setState({ status: 'unavailable', message }),
    );

    return unsubscribe;
  }, [courseId, attempt]);

  const retry = useCallback(() => {
    setAttempt((current) => current + 1);
  }, []);

  return {
    state,
    metadata: state.status === 'ready' ? state.metadata : null,
    retry,
  };
}
