import { useEffect, useState } from 'react';
import { listCourseSeeds } from '../lib/api';
import { countExamples, type ExampleCounts } from '../lib/exampleCounts';

export type CountsState =
  | { status: 'loading' }
  | { status: 'ready'; counts: ExampleCounts }
  | { status: 'unavailable' };

/**
 * Review-status counts for a single course.
 *
 * Used by overview and hub screens that need "how many are waiting for me"
 * without loading the review workflow. Failure is reported as `unavailable`
 * rather than an error: a missing count should never make a course page look
 * broken, and older courses may have no examples recorded at all.
 */
export function useCourseExampleCounts(courseId: string | null): CountsState {
  const [state, setState] = useState<CountsState>({ status: 'loading' });

  useEffect(() => {
    if (!courseId) {
      setState({ status: 'unavailable' });
      return;
    }

    let cancelled = false;
    setState({ status: 'loading' });

    void listCourseSeeds(courseId)
      .then((response) => {
        if (!cancelled) {
          setState({ status: 'ready', counts: countExamples(response.seeds || []) });
        }
      })
      .catch(() => {
        if (!cancelled) {
          setState({ status: 'unavailable' });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [courseId]);

  return state;
}
