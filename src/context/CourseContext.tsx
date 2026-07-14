import { createContext, useContext } from 'react';
import { DEFAULT_COURSE_ID } from '../lib/courseId';

export interface CourseContextValue {
  courseId: string;
}

const CourseContext = createContext<CourseContextValue | null>(null);

export function CourseProvider({
  courseId,
  children,
}: {
  courseId: string;
  children: React.ReactNode;
}) {
  return (
    <CourseContext.Provider value={{ courseId }}>{children}</CourseContext.Provider>
  );
}

/**
 * Returns the validated courseId from the nearest CourseProvider.
 * Falls back to DEFAULT_COURSE_ID for components rendered outside a course route
 * (e.g. Architecture, Not Found).
 */
export function useCourseId(): string {
  return useContext(CourseContext)?.courseId ?? DEFAULT_COURSE_ID;
}
