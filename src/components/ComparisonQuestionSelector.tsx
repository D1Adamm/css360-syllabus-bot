import type { ComparisonRecord } from '../types';

interface ComparisonQuestionSelectorProps {
  records: ComparisonRecord[];
  selectedId: string;
  isRunning?: boolean;
  isRunDisabled?: boolean;
  onSelect: (id: string) => void;
  onRunComparison?: () => void;
}

export function ComparisonQuestionSelector({
  records,
  selectedId,
  isRunning = false,
  isRunDisabled = false,
  onSelect,
  onRunComparison,
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

      {onRunComparison && (
        <div className="comparison-selector__actions">
          <button
            type="button"
            className="comparison-selector__run"
            onClick={onRunComparison}
            disabled={isRunDisabled || isRunning}
            aria-busy={isRunning}
          >
            {isRunning || isRunDisabled ? 'Running comparison…' : 'Run comparison'}
          </button>
        </div>
      )}

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
