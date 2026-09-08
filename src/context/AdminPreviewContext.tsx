import { createContext, useContext } from 'react';
import { isAdminPreview } from './session';
import { useSession } from './SessionContext';

/**
 * Whether the student pages below are an administrator's preview.
 *
 * True exactly when `isAdminPreview` says so for the course in the URL: an
 * administrator who has not joined it. In preview the pages render and request
 * as they do for a student — the same four generation routes for the same
 * course — and nothing submitted is saved: Evaluate sends its rating to the
 * backend's administrator-only preview route, which validates it and stores
 * nothing, and Contribute's form submits nowhere. The shell shows a banner on
 * every page of the course while this is true.
 *
 * Mounted by `CourseRoute` for the student tree only. Anywhere without the
 * provider — the professor and admin trees, and a page rendered on its own in a
 * test — reads false, so nothing changes for them.
 */
const AdminPreviewContext = createContext(false);

export function AdminPreviewProvider({
  courseId,
  children,
}: {
  courseId: string;
  children: React.ReactNode;
}) {
  const { session } = useSession();
  return (
    <AdminPreviewContext.Provider value={isAdminPreview(session, courseId)}>
      {children}
    </AdminPreviewContext.Provider>
  );
}

export function useAdminPreview(): boolean {
  return useContext(AdminPreviewContext);
}
