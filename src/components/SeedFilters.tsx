import {
  ALL_ANSWER_TYPES,
  ALL_CATEGORIES,
  ALL_DIFFICULTIES,
  type AnswerTypeFilter,
  type SortOption,
} from '../utils/seedDataUtils';

interface SeedFiltersProps {
  searchQuery: string;
  onSearchChange: (value: string) => void;
  categories: string[];
  selectedCategory: string;
  onCategoryChange: (category: string) => void;
  selectedDifficulty: string;
  onDifficultyChange: (difficulty: string) => void;
  selectedAnswerType: AnswerTypeFilter;
  onAnswerTypeChange: (answerType: AnswerTypeFilter) => void;
  sortBy: SortOption;
  onSortChange: (sort: SortOption) => void;
  onExportFilteredJson: () => void;
  onExportFilteredJsonl: () => void;
  onExportCompleteJsonl: () => void;
  onExportUserJsonl?: () => void;
  userSeedCount?: number;
}

const DIFFICULTIES = ['Easy', 'Medium', 'Hard'] as const;

const ANSWER_TYPE_OPTIONS: AnswerTypeFilter[] = [
  ALL_ANSWER_TYPES,
  'Directly answered',
  'Not directly answered',
];

const SORT_OPTIONS: { value: SortOption; label: string }[] = [
  { value: 'id-asc', label: 'ID ascending' },
  { value: 'category-asc', label: 'Category A–Z' },
  { value: 'difficulty', label: 'Difficulty' },
  { value: 'question-asc', label: 'Question A–Z' },
];

export function SeedFilters({
  searchQuery,
  onSearchChange,
  categories,
  selectedCategory,
  onCategoryChange,
  selectedDifficulty,
  onDifficultyChange,
  selectedAnswerType,
  onAnswerTypeChange,
  sortBy,
  onSortChange,
  onExportFilteredJson,
  onExportFilteredJsonl,
  onExportCompleteJsonl,
  onExportUserJsonl,
  userSeedCount = 0,
}: SeedFiltersProps) {
  const searchId = 'seed-dataset-search';
  const sortId = 'seed-dataset-sort';

  return (
    <section className="seed-filters" aria-label="Search, filter, sort, and export seed data">
      <div className="seed-filters__search">
        <label htmlFor={searchId} className="seed-filters__label">
          Search examples
        </label>
        <input
          id={searchId}
          type="search"
          className="seed-filters__input"
          placeholder="Search by question, answer, category, or source section…"
          value={searchQuery}
          onChange={(event) => onSearchChange(event.target.value)}
        />
      </div>

      <div className="seed-filters__row">
        <div className="seed-filters__field">
          <label htmlFor="seed-category-filter" className="seed-filters__label">
            Category
          </label>
          <select
            id="seed-category-filter"
            className="seed-filters__select"
            value={selectedCategory}
            onChange={(event) => onCategoryChange(event.target.value)}
          >
            <option value={ALL_CATEGORIES}>{ALL_CATEGORIES}</option>
            {categories.map((category) => (
              <option key={category} value={category}>
                {category}
              </option>
            ))}
          </select>
        </div>

        <div className="seed-filters__field">
          <label htmlFor="seed-difficulty-filter" className="seed-filters__label">
            Difficulty
          </label>
          <select
            id="seed-difficulty-filter"
            className="seed-filters__select"
            value={selectedDifficulty}
            onChange={(event) => onDifficultyChange(event.target.value)}
          >
            <option value={ALL_DIFFICULTIES}>{ALL_DIFFICULTIES}</option>
            {DIFFICULTIES.map((difficulty) => (
              <option key={difficulty} value={difficulty}>
                {difficulty}
              </option>
            ))}
          </select>
        </div>

        <div className="seed-filters__field">
          <label htmlFor="seed-answer-type-filter" className="seed-filters__label">
            Answer type
          </label>
          <select
            id="seed-answer-type-filter"
            className="seed-filters__select"
            value={selectedAnswerType}
            onChange={(event) => onAnswerTypeChange(event.target.value as AnswerTypeFilter)}
          >
            {ANSWER_TYPE_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>

        <div className="seed-filters__field">
          <label htmlFor={sortId} className="seed-filters__label">
            Sort by
          </label>
          <select
            id={sortId}
            className="seed-filters__select"
            value={sortBy}
            onChange={(event) => onSortChange(event.target.value as SortOption)}
          >
            {SORT_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="seed-filters__export">
        <span className="seed-filters__label" id="seed-export-label">
          Export
        </span>
        <div className="seed-filters__export-buttons" role="group" aria-labelledby="seed-export-label">
          <button
            type="button"
            className="seed-filters__export-button"
            onClick={onExportFilteredJson}
          >
            Export filtered results as JSON
          </button>
          <button
            type="button"
            className="seed-filters__export-button"
            onClick={onExportFilteredJsonl}
          >
            Export filtered results as JSONL
          </button>
          <button
            type="button"
            className="seed-filters__export-button"
            onClick={onExportCompleteJsonl}
          >
            Export complete dataset as JSONL
          </button>
          {onExportUserJsonl && (
            <button
              type="button"
              className="seed-filters__export-button"
              onClick={onExportUserJsonl}
              disabled={userSeedCount === 0}
              title={
                userSeedCount === 0
                  ? 'Create examples in the Seed Data Builder to enable this export.'
                  : undefined
              }
            >
              Export user-created examples as JSONL
            </button>
          )}
        </div>
        {onExportUserJsonl && userSeedCount === 0 && (
          <p className="seed-filters__export-hint">
            No user-created examples yet. Use the Seed Data Builder to add examples for
            user-only export.
          </p>
        )}
      </div>
    </section>
  );
}
