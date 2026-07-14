import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { CourseIndexRedirect, CourseRoute } from './components/CourseRoute';
import { Layout } from './components/Layout';
import { LegacyCourseRedirect } from './components/LegacyCourseRedirect';
import { ScrollToTop } from './components/ScrollToTop';
import { ArchitecturePage } from './pages/ArchitecturePage';
import { ComparisonPage } from './pages/ComparisonPage';
import { CreateCoursePage } from './pages/CreateCoursePage';
import { EvaluationPage } from './pages/EvaluationPage';
import { HomePage } from './pages/HomePage';
import { NotFoundPage } from './pages/NotFoundPage';
import { ResultsPage } from './pages/ResultsPage';
import { SeedBuilderPage } from './pages/SeedBuilderPage';
import { SeedDatasetPage } from './pages/SeedDatasetPage';
import { SyllabusPage } from './pages/SyllabusPage';

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/architecture" element={<ArchitecturePage />} />
        <Route path="/create-course" element={<CreateCoursePage />} />

        {/* Legacy routes → default course (temporary backward compatibility) */}
        <Route path="/" element={<LegacyCourseRedirect segment="home" />} />
        <Route path="/home" element={<LegacyCourseRedirect segment="home" />} />
        <Route path="/syllabus" element={<LegacyCourseRedirect segment="syllabus" />} />
        <Route path="/seeds" element={<LegacyCourseRedirect segment="seeds" />} />
        <Route path="/seed-builder" element={<LegacyCourseRedirect segment="seeds" />} />
        <Route path="/dataset" element={<LegacyCourseRedirect segment="dataset" />} />
        <Route path="/compare" element={<LegacyCourseRedirect segment="compare" />} />
        <Route path="/evaluate" element={<LegacyCourseRedirect segment="evaluate" />} />
        <Route path="/results" element={<LegacyCourseRedirect segment="results" />} />

        <Route path="/course/:courseId" element={<CourseRoute />}>
          <Route index element={<CourseIndexRedirect />} />
          <Route path="home" element={<HomePage />} />
          <Route path="syllabus" element={<SyllabusPage />} />
          <Route path="seeds" element={<SeedBuilderPage />} />
          <Route path="dataset" element={<SeedDatasetPage />} />
          <Route path="compare" element={<ComparisonPage />} />
          <Route path="evaluate" element={<EvaluationPage />} />
          <Route path="results" element={<ResultsPage />} />
          <Route path="*" element={<Navigate to="home" replace />} />
        </Route>

        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}

function App() {
  return (
    <BrowserRouter>
      <ScrollToTop />
      <AppRoutes />
    </BrowserRouter>
  );
}

export default App;
