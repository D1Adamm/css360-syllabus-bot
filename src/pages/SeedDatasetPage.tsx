import { useCallback, useMemo, useState } from 'react';
import { DatasetStats } from '../components/DatasetStats';
import { PageHeader } from '../components/PageHeader';
import { ResultsCount } from '../components/ResultsCount';
import { SeedCard } from '../components/SeedCard';
import { SeedFilters } from '../components/SeedFilters';
import { useCourseId } from '../context/CourseContext';
import { useSeedExamples } from '../hooks/useSeedExamples';
import {
  exportCompleteJsonl,
  exportFilteredJson,
  exportFilteredJsonl,
} from '../utils/exportData';
import {
  ALL_ANSWER_TYPES,
  ALL_CATEGORIES,
  ALL_DIFFICULTIES,
  type AnswerTypeFilter,
  calculateStatistics,
  filterSeeds,
  getUniqueCategories,
  type SortOption,
} from '../utils/seedDataUtils';

export function SeedDatasetPage() {
  const courseId = useCourseId();
  const { seeds, loading, error, deleteSeed } = useSeedExamples();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState(ALL_CATEGORIES);
  const [selectedDifficulty, setSelectedDifficulty] = useState(ALL_DIFFICULTIES);
  const [selectedAnswerType, setSelectedAnswerType] = useState<AnswerTypeFilter>(ALL_ANSWER_TYPES);
  const [sortBy, setSortBy] = useState<SortOption>('id-asc');

  const categories = useMemo(() => getUniqueCategories(seeds), [seeds]);
  const stats = useMemo(() => calculateStatistics(seeds), [seeds]);

  const filteredSeeds = useMemo(
    () =>
      filterSeeds(seeds, {
        searchQuery,
        category: selectedCategory,
        difficulty: selectedDifficulty,
        answerType: selectedAnswerType,
        sortBy,
      }),
    [seeds, searchQuery, selectedCategory, selectedDifficulty, selectedAnswerType, sortBy],
  );

  const handleDeleteSeed = useCallback(
    async (id: string) => {
      await deleteSeed(id);
    },
    [deleteSeed],
  );

  function clearFilters() {
    setSearchQuery('');
    setSelectedCategory(ALL_CATEGORIES);
    setSelectedDifficulty(ALL_DIFFICULTIES);
    setSelectedAnswerType(ALL_ANSWER_TYPES);
    setSortBy('id-asc');
  }

  return (
    <>
      <PageHeader
        title="Seed Dataset"
        description="Browse seed examples created for this course. These pairs can support future fine-tuning experiments for a course-specific syllabus assistant."
      />

      <aside className="dataset-notice" aria-label="Dataset notice">
        <p>
          <strong>Course-specific dataset:</strong> examples shown here come only from Firebase{' '}
          <code>courses/{courseId}/seedExamples</code>. Different courses do not share seed
          examples.
        </p>
      </aside>

      {error && (
        <p className="seed-builder-status seed-builder-status--error" role="alert">
          Could not load seed examples for this course: {error}
        </p>
      )}

      {loading ? (
        <p className="seed-builder-status" role="status" aria-live="polite">
          Loading seed examples for this course…
        </p>
      ) : seeds.length === 0 ? (
        <section className="dataset-empty" aria-live="polite">
          <h2 className="dataset-empty__title">No seed examples yet</h2>
          <p className="dataset-empty__text">
            No seed examples have been created for this course yet.
          </p>
        </section>
      ) : (
        <>
          <DatasetStats stats={stats} />

          <SeedFilters
            searchQuery={searchQuery}
            onSearchChange={setSearchQuery}
            categories={categories}
            selectedCategory={selectedCategory}
            onCategoryChange={setSelectedCategory}
            selectedDifficulty={selectedDifficulty}
            onDifficultyChange={setSelectedDifficulty}
            selectedAnswerType={selectedAnswerType}
            onAnswerTypeChange={setSelectedAnswerType}
            sortBy={sortBy}
            onSortChange={setSortBy}
            onExportFilteredJson={() => exportFilteredJson(filteredSeeds)}
            onExportFilteredJsonl={() => exportFilteredJsonl(filteredSeeds)}
            onExportCompleteJsonl={() => exportCompleteJsonl(seeds)}
          />

          <ResultsCount resultCount={filteredSeeds.length} totalCount={seeds.length} />

          {filteredSeeds.length === 0 ? (
            <section className="dataset-empty" aria-live="polite">
              <h2 className="dataset-empty__title">No matching seed examples</h2>
              <p className="dataset-empty__text">
                Try a different search term or filter combination. You can also clear all
                filters to browse the full dataset.
              </p>
              <button type="button" className="dataset-empty__button" onClick={clearFilters}>
                Clear filters
              </button>
            </section>
          ) : (
            <section className="seed-list" aria-label="Seed examples" aria-live="polite">
              {filteredSeeds.map((seed) => (
                <SeedCard key={seed.id} seed={seed} onDelete={handleDeleteSeed} />
              ))}
            </section>
          )}
        </>
      )}
    </>
  );
}
