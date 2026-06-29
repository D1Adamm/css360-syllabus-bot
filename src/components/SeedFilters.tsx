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

      <div className="seed-filters__group">
        <span className="seed-filters__label" id="seed-category-filter-label">
          Category
        </span>
        <div
          className="seed-filters__controls"
          role="group"
          aria-labelledby="seed-category-filter-label"
        >
          <button
            type="button"
            className={`seed-filters__button${
              selectedCategory === ALL_CATEGORIES ? ' seed-filters__button--active' : ''
            }`}
            aria-pressed={selectedCategory === ALL_CATEGORIES}
            onClick={() => onCategoryChange(ALL_CATEGORIES)}
          >
            {ALL_CATEGORIES}
          </button>
          {categories.map((category) => (
            <button
              key={category}
              type="button"
              className={`seed-filters__button${
                selectedCategory === category ? ' seed-filters__button--active' : ''
              }`}
              aria-pressed={selectedCategory === category}
              onClick={() => onCategoryChange(category)}
            >
              {category}
            </button>
          ))}
        </div>
      </div>

      <div className="seed-filters__row">
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
        </div>
      </div>
    </section>
  );
}
