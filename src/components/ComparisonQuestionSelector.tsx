import type { ComparisonRecord } from '../types';

interface ComparisonQuestionSelectorProps {
  records: ComparisonRecord[];
  selectedId: string;
  onSelect: (id: string) => void;
}

export function ComparisonQuestionSelector({
  records,
  selectedId,
  onSelect,
}: ComparisonQuestionSelectorProps) {
  const selected = records.find((record) => record.id === selectedId) ?? records[0];

  return (
    <section className="comparison-selector" aria-labelledby="comparison-selector-title">
      <h2 id="comparison-selector-title" className="comparison-selector__title">
        Select a question
      </h2>

      <label htmlFor="comparison-question-select" className="comparison-selector__label">
        Predefined syllabus questions
      </label>
      <select
        id="comparison-question-select"
        className="comparison-selector__select"
        value={selectedId}
        onChange={(event) => onSelect(event.target.value)}
      >
        {records.map((record) => (
          <option key={record.id} value={record.id}>
            {record.question}
          </option>
        ))}
      </select>

      {selected && (
        <dl className="comparison-selector__meta">
          <div className="comparison-selector__meta-row">
            <dt>Category</dt>
            <dd>{selected.category}</dd>
          </div>
          <div className="comparison-selector__meta-row">
            <dt>Relevant syllabus section</dt>
            <dd>{selected.relevantSyllabusSection}</dd>
          </div>
        </dl>
      )}
    </section>
  );
}
