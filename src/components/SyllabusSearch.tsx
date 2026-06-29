interface SyllabusSearchProps {
  value: string;
  onChange: (value: string) => void;
  resultCount: number;
  totalCount: number;
}

export function SyllabusSearch({
  value,
  onChange,
  resultCount,
  totalCount,
}: SyllabusSearchProps) {
  const inputId = 'syllabus-search';

  return (
    <div className="syllabus-search">
      <label htmlFor={inputId} className="syllabus-search__label">
        Search topics
      </label>
      <input
        id={inputId}
        type="search"
        className="syllabus-search__input"
        placeholder="Search by title, summary, category, or details…"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        aria-describedby="syllabus-search-hint"
      />
      <p id="syllabus-search-hint" className="syllabus-search__hint">
        Showing {resultCount} of {totalCount} syllabus topics
      </p>
    </div>
  );
}
