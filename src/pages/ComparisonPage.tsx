import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import comparisonData from '../data/comparisonData.json';
import { ComparisonExplanation } from '../components/ComparisonExplanation';
import { ComparisonQuestionSelector } from '../components/ComparisonQuestionSelector';
import { CustomQuestionMatcher } from '../components/CustomQuestionMatcher';
import { ModelResponseCard } from '../components/ModelResponseCard';
import { PageHeader } from '../components/PageHeader';
import type { ComparisonRecord } from '../types';

const records = comparisonData as ComparisonRecord[];

const MODEL_CARDS = [
  {
    key: 'base',
    modelName: 'Base Model',
    accessDescription: 'No syllabus context and no student-created fine-tuning examples.',
    responseKey: 'baseResponse' as const,
  },
  {
    key: 'rag',
    modelName: 'RAG',
    accessDescription: 'Receives relevant syllabus passages at answer time.',
    responseKey: 'ragResponse' as const,
  },
  {
    key: 'fineTuned',
    modelName: 'Fine-Tuned Model',
    accessDescription: 'Uses response behavior learned from prototype seed examples.',
    responseKey: 'fineTunedResponse' as const,
  },
  {
    key: 'fineTunedRag',
    modelName: 'Fine-Tuned + RAG',
    accessDescription:
      'Uses both learned response behavior and retrieved syllabus context.',
    responseKey: 'fineTunedRagResponse' as const,
  },
];

export function ComparisonPage() {
  const [selectedId, setSelectedId] = useState(records[0]?.id ?? '');

  const selectedRecord = useMemo(
    () => records.find((record) => record.id === selectedId) ?? records[0],
    [selectedId],
  );

  if (!selectedRecord) {
    return null;
  }

  const evaluateHref = `/evaluate?comparison=${selectedRecord.id}`;

  return (
    <>
      <PageHeader
        title="Model Comparison"
        description="Compare how different model approaches may answer the same syllabus question. Each configuration has different access to syllabus content and seed-example training data."
      />

      <aside className="comparison-notice" aria-label="Simulation notice">
        <p>
          <strong>Simulation only:</strong> these responses were written in advance for the
          prototype. No live model is running.
        </p>
      </aside>

      <ComparisonQuestionSelector
        records={records}
        selectedId={selectedRecord.id}
        onSelect={setSelectedId}
      />

      <CustomQuestionMatcher
        records={records}
        onMatch={(recordId) => setSelectedId(recordId)}
        onNoMatch={() => {
          // Keep the current selection; the matcher shows the limitation message.
        }}
      />

      <section className="comparison-grid" aria-label="Model responses">
        {MODEL_CARDS.map((card) => (
          <ModelResponseCard
            key={card.key}
            modelName={card.modelName}
            accessDescription={card.accessDescription}
            response={selectedRecord[card.responseKey]}
          />
        ))}
      </section>

      <ComparisonExplanation notes={selectedRecord.notes} />

      <div className="comparison-actions">
        <Link to={evaluateHref} className="button-link button-link--primary">
          Evaluate these responses
        </Link>
      </div>
    </>
  );
}
