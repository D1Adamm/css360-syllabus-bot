import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import {
  LegacyCourseRedirect,
  LegacyFlatRedirect,
  RoleLanding,
} from './app/LegacyRedirects';
import {
  RequireAdmin,
  RequireAnyPrincipal,
  RequireCourseParticipant,
  RequireCourseStaff,
  RequireStaff,
} from './components/auth/RouteGuards';
import { CourseRoute } from './components/CourseRoute';
import { ScrollToTop } from './components/ScrollToTop';
import { ComparisonRunProvider } from './context/ComparisonRunContext';
import { SessionProvider } from './context/SessionContext';
import { AppShell } from './shell/AppShell';

import { ForbiddenPage } from './pages/ForbiddenPage';
import { NotFoundPage } from './pages/NotFoundPage';
import { StyleguidePage } from './pages/StyleguidePage';

import { AcceptInvitePage } from './pages/auth/AcceptInvitePage';
import { JoinPage } from './pages/auth/JoinPage';
import { LoginPage } from './pages/auth/LoginPage';

import { ComparePage } from './pages/student/ComparePage';
import { ContributePage } from './pages/student/ContributePage';
import { EvaluatePage } from './pages/student/EvaluatePage';
import { StudentCoursesPage } from './pages/student/StudentCoursesPage';
import { StudentHomePage } from './pages/student/StudentHomePage';
import { StudentSyllabusPage } from './pages/student/StudentSyllabusPage';

import { CourseOverviewPage } from './pages/professor/CourseOverviewPage';
import { CreateCoursePage } from './pages/professor/CreateCoursePage';
import { InviteStudentsPage } from './pages/professor/InviteStudentsPage';
import { ProfessorCoursesPage } from './pages/professor/ProfessorCoursesPage';
import { ProfessorModelPage } from './pages/professor/ProfessorModelPage';
import { ProfessorResultsPage } from './pages/professor/ProfessorResultsPage';
import { ProfessorSyllabusPage } from './pages/professor/ProfessorSyllabusPage';
import { ReviewExamplesPage } from './pages/professor/ReviewExamplesPage';

import { AdminCourseDetailPage } from './pages/admin/AdminCourseDetailPage';
import { AdminCoursesPage } from './pages/admin/AdminCoursesPage';
import { AdminExamplesPage } from './pages/admin/AdminExamplesPage';
import { AdminModelsPage } from './pages/admin/AdminModelsPage';
import { AdminOverviewPage } from './pages/admin/AdminOverviewPage';
import { AdminSystemPage } from './pages/admin/AdminSystemPage';
import { AdminTrainingPage } from './pages/admin/AdminTrainingPage';

/**
 * The route tree.
 *
 * Guards wrap each role area. They are navigation, not security: a guard sends
 * the browser to sign-in, the join page, or an explanation, and the backend
 * refuses anything a page inside should not have been able to fetch.
 */
export function AppRoutes() {
  return (
    <Routes>
      {/* Design-system review page, outside the app shell. Development only. */}
      <Route path="/styleguide" element={<StyleguidePage />} />

      <Route element={<AppShell />}>
        <Route path="/" element={<RoleLanding />} />

        {/* ------------------------------ Sessions ----------------------------- */}
        <Route path="/login" element={<LoginPage />} />
        <Route path="/join" element={<JoinPage />} />
        <Route path="/join/:code" element={<JoinPage />} />
        <Route path="/invite/:token" element={<AcceptInvitePage />} />
        <Route path="/forbidden" element={<ForbiddenPage />} />

        {/* ------------------------------ Student ------------------------------ */}
        <Route
          path="/student"
          element={
            <RequireAnyPrincipal>
              <StudentCoursesPage />
            </RequireAnyPrincipal>
          }
        />
        <Route
          path="/student/course/:courseId"
          element={
            <RequireCourseParticipant>
              <CourseRoute />
            </RequireCourseParticipant>
          }
        >
          <Route index element={<StudentHomePage />} />
          <Route path="syllabus" element={<StudentSyllabusPage />} />
          <Route path="contribute" element={<ContributePage />} />
          <Route path="compare" element={<ComparePage />} />
          <Route path="evaluate" element={<EvaluatePage />} />
          <Route path="*" element={<Navigate to="." replace />} />
        </Route>

        {/* ----------------------------- Professor ----------------------------- */}
        <Route path="/professor" element={<Navigate to="/professor/courses" replace />} />
        <Route
          path="/professor/courses"
          element={
            <RequireStaff>
              <ProfessorCoursesPage />
            </RequireStaff>
          }
        />
        <Route
          path="/professor/courses/new"
          element={
            <RequireStaff>
              <CreateCoursePage />
            </RequireStaff>
          }
        />
        {/* The cross-course hubs are gone from navigation; their URLs still
            resolve so existing links do not break. */}
        <Route
          path="/professor/reviews"
          element={<Navigate to="/professor/courses" replace />}
        />
        <Route
          path="/professor/models"
          element={<Navigate to="/professor/courses" replace />}
        />
        <Route
          path="/professor/course/:courseId"
          element={
            <RequireCourseStaff>
              <CourseRoute />
            </RequireCourseStaff>
          }
        >
          <Route index element={<CourseOverviewPage />} />
          <Route path="syllabus" element={<ProfessorSyllabusPage />} />
          <Route path="examples" element={<ReviewExamplesPage />} />
          <Route path="model" element={<ProfessorModelPage />} />
          <Route path="results" element={<ProfessorResultsPage />} />
          <Route path="invite" element={<InviteStudentsPage />} />
          <Route path="*" element={<Navigate to="." replace />} />
        </Route>

        {/* ------------------------------- Admin ------------------------------- */}
        <Route
          path="/admin"
          element={
            <RequireAdmin>
              <AdminOverviewPage />
            </RequireAdmin>
          }
        />
        <Route
          path="/admin/courses"
          element={
            <RequireAdmin>
              <AdminCoursesPage />
            </RequireAdmin>
          }
        />
        <Route
          path="/admin/training"
          element={
            <RequireAdmin>
              <AdminTrainingPage />
            </RequireAdmin>
          }
        />
        <Route
          path="/admin/models"
          element={
            <RequireAdmin>
              <AdminModelsPage />
            </RequireAdmin>
          }
        />
        <Route
          path="/admin/system"
          element={
            <RequireAdmin>
              <AdminSystemPage />
            </RequireAdmin>
          }
        />
        <Route
          path="/admin/courses/:courseId"
          element={
            <RequireAdmin>
              <CourseRoute />
            </RequireAdmin>
          }
        >
          <Route index element={<AdminCourseDetailPage />} />
          <Route path="examples" element={<AdminExamplesPage />} />
          <Route path="*" element={<Navigate to="." replace />} />
        </Route>

        {/* ---------------------- Redirects from old URLs ---------------------- */}
        <Route path="/architecture" element={<Navigate to="/admin/system" replace />} />
        <Route
          path="/create-course"
          element={<Navigate to="/professor/courses/new" replace />}
        />

        <Route
          path="/course/:courseId"
          element={
            <LegacyCourseRedirect student="home" professor="home" admin="course" />
          }
        />
        <Route
          path="/course/:courseId/home"
          element={
            <LegacyCourseRedirect student="home" professor="home" admin="course" />
          }
        />
        <Route
          path="/course/:courseId/syllabus"
          element={
            <LegacyCourseRedirect
              student="syllabus"
              professor="syllabus"
              admin="course"
            />
          }
        />
        <Route
          path="/course/:courseId/seeds"
          element={<LegacyCourseRedirect student="contribute" />}
        />
        <Route
          path="/course/:courseId/compare"
          element={<LegacyCourseRedirect student="compare" />}
        />
        <Route
          path="/course/:courseId/evaluate"
          element={<LegacyCourseRedirect student="evaluate" />}
        />
        <Route
          path="/course/:courseId/review"
          element={<LegacyCourseRedirect professor="examples" />}
        />
        <Route
          path="/course/:courseId/results"
          element={<LegacyCourseRedirect professor="results" />}
        />
        <Route
          path="/course/:courseId/dataset"
          element={<LegacyCourseRedirect admin="examples" />}
        />

        <Route
          path="/home"
          element={<LegacyFlatRedirect student="home" professor="home" admin="course" />}
        />
        <Route
          path="/syllabus"
          element={
            <LegacyFlatRedirect student="syllabus" professor="syllabus" admin="course" />
          }
        />
        <Route path="/seeds" element={<LegacyFlatRedirect student="contribute" />} />
        <Route
          path="/seed-builder"
          element={<LegacyFlatRedirect student="contribute" />}
        />
        <Route path="/compare" element={<LegacyFlatRedirect student="compare" />} />
        <Route path="/evaluate" element={<LegacyFlatRedirect student="evaluate" />} />
        <Route path="/review" element={<LegacyFlatRedirect professor="examples" />} />
        <Route path="/results" element={<LegacyFlatRedirect professor="results" />} />
        <Route path="/dataset" element={<LegacyFlatRedirect admin="examples" />} />

        <Route path="/not-found" element={<NotFoundPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}

function App() {
  return (
    <BrowserRouter>
      <SessionProvider>
        <ComparisonRunProvider>
          <ScrollToTop />
          <AppRoutes />
        </ComparisonRunProvider>
      </SessionProvider>
    </BrowserRouter>
  );
}

export default App;
