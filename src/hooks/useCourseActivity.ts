import { useEffect, useState } from 'react';
import { getCourseActivity } from '../lib/dbApi';
import type { CourseActivity } from '../types';

export type CourseActivityState =
  | { status: 'loading' }
  | { status: 'ready'; activity: CourseActivity }
  | { status: 'unavailable' };

/**
 * Class-wide counts for the student home page.
 *
 * Counts only. The page used to subscribe to every evaluation in the course
 * to show how many there were, which meant every student's browser downloaded
 * every classmate's rating and comment. A participant now never receives
 * another student's rating at all; the number comes from here.
 */
export function useCourseActivity(courseId: string): CourseActivityState {
  const [state, setState] = useState<CourseActivityState>({ status: 'loading' });

  useEffect(() => {
    let cancelled = false;
    setState({ status: 'loading' });
    getCourseActivity(courseId)
      .then((activity) => {
        if (!cancelled) {
          setState({ status: 'ready', activity });
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
