import { createContext, useContext } from 'react';

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
 * Course pages must be rendered under `/course/:courseId/...`.
 */
export function useCourseId(): string {
  const value = useContext(CourseContext);
  if (!value) {
    throw new Error(
      'useCourseId requires a CourseProvider. Open a course route under /course/:courseId/.',
    );
  }
  return value.courseId;
}
