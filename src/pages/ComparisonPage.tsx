import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import comparisonData from '../data/comparisonData.json';
import { ComparisonExplanation } from '../components/ComparisonExplanation';
import { ComparisonQuestionSelector } from '../components/ComparisonQuestionSelector';
import { CustomQuestionMatcher } from '../components/CustomQuestionMatcher';
import { ModelResponseCard } from '../components/ModelResponseCard';
import { PageHeader } from '../components/PageHeader';
import { ApiError, generateBaseModel } from '../lib/api';
import type { ComparisonRecord, ComparisonResponse } from '../types';

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

const DEFAULT_BASE_RESPONSE: ComparisonResponse = {
  text: '',
  grounding: 'Low',
  simulated: false,
};

type BaseModelState =
  | { status: 'idle' }
  | { status: 'loading'; question: string }
  | { status: 'success'; question: string; response: ComparisonResponse }
  | { status: 'error'; question: string; message: string };

function getBaseModelErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }

  return 'The Base Model response could not be loaded.';
}

export function ComparisonPage() {
  const [selectedId, setSelectedId] = useState(records[0]?.id ?? '');
  const [activeQuestion, setActiveQuestion] = useState(records[0]?.question ?? '');
  const [baseModelState, setBaseModelState] = useState<BaseModelState>({ status: 'idle' });

  const selectedRecord = useMemo(
    () => records.find((record) => record.id === selectedId) ?? records[0],
    [selectedId],
  );

  const loadBaseModelResponse = useCallback(async (question: string) => {
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion) {
      return;
    }

    setBaseModelState({ status: 'loading', question: trimmedQuestion });

    try {
      const result = await generateBaseModel(trimmedQuestion);

      setBaseModelState({
        status: 'success',
        question: trimmedQuestion,
        response: {
          text: result.answer,
          grounding: 'Low',
          simulated: false,
        },
      });
    } catch (error) {
      setBaseModelState({
        status: 'error',
        question: trimmedQuestion,
        message: getBaseModelErrorMessage(error),
      });
    }
  }, []);

  useEffect(() => {
    void loadBaseModelResponse(activeQuestion);
  }, [activeQuestion, loadBaseModelResponse]);

  if (!selectedRecord) {
    return null;
  }

  const evaluateHref = `/evaluate?comparison=${selectedRecord.id}`;
  const isBaseModelLoading = baseModelState.status === 'loading';
  const baseModelError = baseModelState.status === 'error' ? baseModelState.message : null;
  const baseModelResponse =
    baseModelState.status === 'success' ? baseModelState.response : DEFAULT_BASE_RESPONSE;

  return (
    <>
      <PageHeader
        title="Model Comparison"
        description="Compare how different model approaches may answer the same syllabus question. Each configuration has different access to syllabus content and seed-example training data."
      />

      <aside className="comparison-notice" aria-label="Simulation notice">
        <p>
          <strong>Hybrid prototype:</strong> the Base Model response is live from the local
          FastAPI backend. RAG, Fine-Tuned, and Fine-Tuned + RAG responses remain simulated.
        </p>
      </aside>

      <ComparisonQuestionSelector
        records={records}
        selectedId={selectedRecord.id}
        onSelect={(id) => {
          setSelectedId(id);
          const record = records.find((item) => item.id === id);
          if (record) {
            setActiveQuestion(record.question);
          }
        }}
      />

      <CustomQuestionMatcher
        records={records}
        onMatch={(recordId) => setSelectedId(recordId)}
        onNoMatch={() => {
          // Keep the current selection for simulated responses.
        }}
        onQuestionSubmit={setActiveQuestion}
      />

      <section className="comparison-grid" aria-label="Model responses">
        {MODEL_CARDS.map((card) => {
          if (card.key === 'base') {
            return (
              <ModelResponseCard
                key={card.key}
                modelName={card.modelName}
                accessDescription={card.accessDescription}
                response={baseModelResponse}
                isLoading={isBaseModelLoading}
                error={baseModelError}
              />
            );
          }

          return (
            <ModelResponseCard
              key={card.key}
              modelName={card.modelName}
              accessDescription={card.accessDescription}
              response={selectedRecord[card.responseKey]}
            />
          );
        })}
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
