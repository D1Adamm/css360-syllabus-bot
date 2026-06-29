import { useMemo, useState } from 'react';
import seedData from '../data/seedData.json';
import { DatasetStats } from '../components/DatasetStats';
import { PageHeader } from '../components/PageHeader';
import { ResultsCount } from '../components/ResultsCount';
import { SeedCard } from '../components/SeedCard';
import { SeedFilters } from '../components/SeedFilters';
import type { SeedExample } from '../types';
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

const allSeeds = seedData as SeedExample[];

export function SeedDatasetPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState(ALL_CATEGORIES);
  const [selectedDifficulty, setSelectedDifficulty] = useState(ALL_DIFFICULTIES);
  const [selectedAnswerType, setSelectedAnswerType] = useState<AnswerTypeFilter>(ALL_ANSWER_TYPES);
  const [sortBy, setSortBy] = useState<SortOption>('id-asc');

  const categories = useMemo(() => getUniqueCategories(allSeeds), []);
  const stats = useMemo(() => calculateStatistics(allSeeds), []);

  const filteredSeeds = useMemo(
    () =>
      filterSeeds(allSeeds, {
        searchQuery,
        category: selectedCategory,
        difficulty: selectedDifficulty,
        answerType: selectedAnswerType,
        sortBy,
      }),
    [searchQuery, selectedCategory, selectedDifficulty, selectedAnswerType, sortBy],
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
        description="Browse prototype question-and-answer examples derived from the syllabus. These pairs demonstrate the kind of training data students could create for fine-tuning a course assistant."
      />

      <aside className="dataset-notice" aria-label="Prototype dataset notice">
        <p>
          <strong>Prototype dataset:</strong> these examples were generated for the demo
          and were not submitted by students.
        </p>
      </aside>

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
        onExportCompleteJsonl={() => exportCompleteJsonl(allSeeds)}
      />

      <ResultsCount resultCount={filteredSeeds.length} totalCount={allSeeds.length} />

      {filteredSeeds.length === 0 ? (
        <section className="dataset-empty" aria-live="polite">
          <h2 className="dataset-empty__title">No matching seed examples</h2>
          <p className="dataset-empty__text">
            Try a different search term or filter combination. You can also clear all
            filters to browse the full prototype dataset.
          </p>
          <button type="button" className="dataset-empty__button" onClick={clearFilters}>
            Clear filters
          </button>
        </section>
      ) : (
        <section className="seed-list" aria-label="Seed examples" aria-live="polite">
          {filteredSeeds.map((seed) => (
            <SeedCard key={seed.id} seed={seed} />
          ))}
        </section>
      )}
    </>
  );
}
