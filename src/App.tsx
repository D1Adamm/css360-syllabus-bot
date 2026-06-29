import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { ScrollToTop } from './components/ScrollToTop';
import { HomePage } from './pages/HomePage';
import { SyllabusPage } from './pages/SyllabusPage';
import { SeedBuilderPage } from './pages/SeedBuilderPage';
import { SeedDatasetPage } from './pages/SeedDatasetPage';
import { ComparisonPage } from './pages/ComparisonPage';
import { EvaluationPage } from './pages/EvaluationPage';
import { ResultsPage } from './pages/ResultsPage';
import { ArchitecturePage } from './pages/ArchitecturePage';
import { NotFoundPage } from './pages/NotFoundPage';

function App() {
  return (
    <BrowserRouter>
      <ScrollToTop />
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/syllabus" element={<SyllabusPage />} />
          <Route path="/seed-builder" element={<SeedBuilderPage />} />
          <Route path="/dataset" element={<SeedDatasetPage />} />
          <Route path="/compare" element={<ComparisonPage />} />
          <Route path="/evaluate" element={<EvaluationPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/architecture" element={<ArchitecturePage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
