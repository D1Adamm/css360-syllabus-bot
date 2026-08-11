import { useCallback, useEffect, useState } from 'react';
import {
  createCourseModelRequest,
  subscribeToCourseModelRequest,
} from '../lib/courseModelRequestDb';
import type { CourseModelRequest } from '../types';

export type CourseModelRequestState =
  | { status: 'loading' }
  /** Read successfully; this course has never requested a model. */
  | { status: 'none' }
  | { status: 'ready'; request: CourseModelRequest }
  /** Could not read. Says nothing about whether a request exists. */
  | { status: 'unavailable'; message: string };

export interface UseCourseModelRequestResult {
  state: CourseModelRequestState;
  /** True while a submission is in flight. */
  submitting: boolean;
  /** Set when the last submission failed. */
  submitError: string | null;
  submit: (approvedExampleCount: number) => Promise<void>;
  clearSubmitError: () => void;
}

/**
 * The model request for one course.
 *
 * `none` and `unavailable` stay separate for the same reason they do
 * everywhere else in this application: "nothing has been requested" and "we
 * could not check" lead to different things being offered to a professor, and
 * guessing between them would put a Request button in front of someone whose
 * request is already running.
 *
 * The subscription is live, so the button disappears the moment the write
 * lands — no optimistic local state to reconcile.
 */
export function useCourseModelRequest(
  courseId: string | null,
): UseCourseModelRequestResult {
  const [state, setState] = useState<CourseModelRequestState>({ status: 'loading' });
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => {
    if (!courseId) {
      setState({ status: 'none' });
      return;
    }

    setState({ status: 'loading' });

    const unsubscribe = subscribeToCourseModelRequest(
      courseId,
      (request) => setState(request ? { status: 'ready', request } : { status: 'none' }),
      (message) => setState({ status: 'unavailable', message }),
    );

    return unsubscribe;
  }, [courseId]);

  const submit = useCallback(
    async (approvedExampleCount: number) => {
      if (!courseId) {
        return;
      }

      setSubmitting(true);
      setSubmitError(null);

      try {
        await createCourseModelRequest(courseId, approvedExampleCount);
      } catch (error) {
        // A duplicate is not really a failure — the subscription will deliver
        // the request that won — but anything else needs saying.
        setSubmitError(
          error instanceof Error && error.name === 'DuplicateModelRequestError'
            ? 'A request for this course is already in progress.'
            : "We couldn't send that request. Try again in a moment.",
        );
      } finally {
        setSubmitting(false);
      }
    },
    [courseId],
  );

  const clearSubmitError = useCallback(() => setSubmitError(null), []);

  return { state, submitting, submitError, submit, clearSubmitError };
}
