interface CategoryFilterProps {
  categories: string[];
  selectedCategory: string;
  onChange: (category: string) => void;
  categoryCounts: Record<string, number>;
}

const ALL_CATEGORIES = 'All categories';

export function CategoryFilter({
  categories,
  selectedCategory,
  onChange,
  categoryCounts,
}: CategoryFilterProps) {
  return (
    <div className="category-filter">
      <span className="category-filter__label" id="category-filter-label">
        Filter by category
      </span>
      <div
        className="category-filter__controls"
        role="group"
        aria-labelledby="category-filter-label"
      >
        <button
          type="button"
          className={`category-filter__button${
            selectedCategory === ALL_CATEGORIES
              ? ' category-filter__button--active'
              : ''
          }`}
          aria-pressed={selectedCategory === ALL_CATEGORIES}
          onClick={() => onChange(ALL_CATEGORIES)}
        >
          {ALL_CATEGORIES}
          <span className="category-filter__count">{categoryCounts[ALL_CATEGORIES]}</span>
        </button>
        {categories.map((category) => (
          <button
            key={category}
            type="button"
            className={`category-filter__button${
              selectedCategory === category ? ' category-filter__button--active' : ''
            }`}
            aria-pressed={selectedCategory === category}
            onClick={() => onChange(category)}
          >
            {category}
            <span className="category-filter__count">{categoryCounts[category] ?? 0}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

export { ALL_CATEGORIES };
