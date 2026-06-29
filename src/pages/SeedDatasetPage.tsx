import { useCallback, useMemo, useState } from 'react';
import seedData from '../data/seedData.json';
import { DatasetStats } from '../components/DatasetStats';
import { PageHeader } from '../components/PageHeader';
import { ResultsCount } from '../components/ResultsCount';
import { SeedCard } from '../components/SeedCard';
import { SeedFilters } from '../components/SeedFilters';
import { useLocalStorage } from '../hooks/useLocalStorage';
import type { SeedExample } from '../types';
import {
  exportCompleteJsonl,
  exportFilteredJson,
  exportFilteredJsonl,
  exportUserSeedsJsonl,
} from '../utils/exportData';
import {
  ALL_ANSWER_TYPES,
  ALL_CATEGORIES,
  ALL_DIFFICULTIES,
  type AnswerTypeFilter,
  calculateStatistics,
  combinePrototypeAndUserSeeds,
  filterSeeds,
  getUniqueCategories,
  type SortOption,
  USER_SEEDS_STORAGE_KEY,
  isSeedExampleArray,
} from '../utils/seedDataUtils';

const prototypeSeeds = seedData as SeedExample[];

export function SeedDatasetPage() {
  const [userSeeds, setUserSeeds] = useLocalStorage<SeedExample[]>(
    USER_SEEDS_STORAGE_KEY,
    [],
    isSeedExampleArray,
  );
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState(ALL_CATEGORIES);
  const [selectedDifficulty, setSelectedDifficulty] = useState(ALL_DIFFICULTIES);
  const [selectedAnswerType, setSelectedAnswerType] = useState<AnswerTypeFilter>(ALL_ANSWER_TYPES);
  const [sortBy, setSortBy] = useState<SortOption>('id-asc');

  const allSeeds = useMemo(
    () => combinePrototypeAndUserSeeds(prototypeSeeds, userSeeds),
    [userSeeds],
  );

  const categories = useMemo(() => getUniqueCategories(allSeeds), [allSeeds]);
  const stats = useMemo(() => calculateStatistics(allSeeds), [allSeeds]);

  const filteredSeeds = useMemo(
    () =>
      filterSeeds(allSeeds, {
        searchQuery,
        category: selectedCategory,
        difficulty: selectedDifficulty,
        answerType: selectedAnswerType,
        sortBy,
      }),
    [allSeeds, searchQuery, selectedCategory, selectedDifficulty, selectedAnswerType, sortBy],
  );

  const handleDeleteUserSeed = useCallback(
    (id: string) => {
      setUserSeeds((current) => current.filter((seed) => seed.id !== id));
    },
    [setUserSeeds],
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
        description="Browse prototype and user-created question-and-answer examples derived from the syllabus. These pairs demonstrate the kind of training data students could create for fine-tuning a course assistant."
      />

      <aside className="dataset-notice" aria-label="Dataset notice">
        <p>
          <strong>Combined dataset:</strong> this page shows {prototypeSeeds.length} prototype
          examples and {userSeeds.length} user-created example
          {userSeeds.length === 1 ? '' : 's'} ({allSeeds.length} total). Prototype examples
          cannot be deleted here.
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
        onExportUserJsonl={() => exportUserSeedsJsonl(userSeeds)}
        userSeedCount={userSeeds.length}
      />

      <ResultsCount resultCount={filteredSeeds.length} totalCount={allSeeds.length} />

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
            <SeedCard
              key={seed.id}
              seed={seed}
              onDelete={seed.origin === 'user' ? handleDeleteUserSeed : undefined}
            />
          ))}
        </section>
      )}
    </>
  );
}
