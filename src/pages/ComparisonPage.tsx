import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import comparisonData from '../data/comparisonData.json';
import { ComparisonExplanation } from '../components/ComparisonExplanation';
import { ComparisonQuestionSelector } from '../components/ComparisonQuestionSelector';
import { CustomQuestionMatcher } from '../components/CustomQuestionMatcher';
import { ModelResponseCard } from '../components/ModelResponseCard';
import { PageHeader } from '../components/PageHeader';
import { ApiError, generateBaseModel, generateRag } from '../lib/api';
import type { ComparisonRecord, ComparisonResponse } from '../types';
import {
  type ComparisonMode,
  SIMULATED_UNAVAILABLE_MESSAGE,
  resolveSimulatedRecord,
} from '../utils/comparisonPageState';

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

const DEFAULT_RAG_RESPONSE: ComparisonResponse = {
  text: '',
  grounding: 'High',
  simulated: false,
};

type LiveModelState =
  | { status: 'idle' }
  | { status: 'loading'; question: string }
  | { status: 'success'; question: string; response: ComparisonResponse }
  | { status: 'error'; question: string; message: string };

type RagModelState =
  | { status: 'idle' }
  | { status: 'loading'; question: string }
  | { status: 'success'; question: string; response: ComparisonResponse; sources: string[] }
  | { status: 'error'; question: string; message: string };

function getApiErrorMessage(error: unknown, fallbackMessage: string): string {
  if (error instanceof ApiError) {
    return error.message;
  }

  return fallbackMessage;
}

export function ComparisonPage() {
  const [selectedId, setSelectedId] = useState(records[0]?.id ?? '');
  const [activeQuestion, setActiveQuestion] = useState(records[0]?.question ?? '');
  const [comparisonMode, setComparisonMode] = useState<ComparisonMode>('predefined');
  const [matchedRecordId, setMatchedRecordId] = useState<string | null>(null);
  const [baseModelState, setBaseModelState] = useState<LiveModelState>({ status: 'idle' });
  const [ragModelState, setRagModelState] = useState<RagModelState>({ status: 'idle' });
  const [customMatcherResetKey, setCustomMatcherResetKey] = useState(0);

  const selectedRecord = useMemo(
    () => records.find((record) => record.id === selectedId) ?? records[0],
    [selectedId],
  );

  const matchedRecord = useMemo(
    () =>
      matchedRecordId
        ? (records.find((record) => record.id === matchedRecordId) ?? null)
        : null,
    [matchedRecordId],
  );

  const simulatedRecord = useMemo(() => {
    if (!selectedRecord) {
      return null;
    }

    return resolveSimulatedRecord(comparisonMode, selectedRecord, matchedRecord);
  }, [comparisonMode, matchedRecord, selectedRecord]);

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
        message: getApiErrorMessage(
          error,
          'The Base Model response could not be loaded.',
        ),
      });
    }
  }, []);

  const loadRagModelResponse = useCallback(async (question: string) => {
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion) {
      return;
    }

    setRagModelState({ status: 'loading', question: trimmedQuestion });

    try {
      const result = await generateRag(trimmedQuestion);

      setRagModelState({
        status: 'success',
        question: trimmedQuestion,
        response: {
          text: result.answer,
          grounding: 'High',
          simulated: false,
        },
        sources: result.sources.map((source) => source.section),
      });
    } catch (error) {
      setRagModelState({
        status: 'error',
        question: trimmedQuestion,
        message: getApiErrorMessage(error, 'The RAG response could not be loaded.'),
      });
    }
  }, []);

  useEffect(() => {
    void Promise.all([
      loadBaseModelResponse(activeQuestion),
      loadRagModelResponse(activeQuestion),
    ]);
  }, [activeQuestion, loadBaseModelResponse, loadRagModelResponse]);

  if (!selectedRecord) {
    return null;
  }

  const evaluateHref = `/evaluate?comparison=${simulatedRecord?.id ?? selectedRecord.id}`;
  const isBaseModelLoading = baseModelState.status === 'loading';
  const isRagModelLoading = ragModelState.status === 'loading';
  const isLiveRequestInFlight = isBaseModelLoading || isRagModelLoading;
  const baseModelError = baseModelState.status === 'error' ? baseModelState.message : null;
  const ragModelError = ragModelState.status === 'error' ? ragModelState.message : null;
  const baseModelResponse =
    baseModelState.status === 'success' ? baseModelState.response : DEFAULT_BASE_RESPONSE;
  const ragModelResponse =
    ragModelState.status === 'success' ? ragModelState.response : DEFAULT_RAG_RESPONSE;
  const ragSources = ragModelState.status === 'success' ? ragModelState.sources : [];
  const showSimulatedUnavailable = comparisonMode === 'custom-unmatched';
  const explanationNotes =
    comparisonMode === 'custom-unmatched'
      ? 'Fine-Tuned and Fine-Tuned + RAG simulated responses are only available for predefined questions or custom questions that closely match a predefined example.'
      : (simulatedRecord?.notes ?? selectedRecord.notes);

  return (
    <>
      <PageHeader
        title="Model Comparison"
        description="Compare how different model approaches may answer the same syllabus question. Each configuration has different access to syllabus content and seed-example training data."
      />

      <aside className="comparison-notice" aria-label="Simulation notice">
        <p>
          <strong>Hybrid prototype:</strong> the Base Model and RAG responses are live from the
          local FastAPI backend. Fine-Tuned and Fine-Tuned + RAG responses remain simulated.
        </p>
      </aside>

      <p className="comparison-active-question" role="status">
        <strong>Active question:</strong> {activeQuestion}
      </p>

      <ComparisonQuestionSelector
        records={records}
        selectedId={selectedRecord.id}
        onSelect={(id) => {
          setSelectedId(id);
          const record = records.find((item) => item.id === id);
          if (record) {
            setActiveQuestion(record.question);
          }
          setComparisonMode('predefined');
          setMatchedRecordId(null);
          setCustomMatcherResetKey((current) => current + 1);
        }}
      />

      <CustomQuestionMatcher
        records={records}
        isSubmitting={isLiveRequestInFlight}
        resetKey={customMatcherResetKey}
        onMatch={(recordId) => {
          setComparisonMode('custom-matched');
          setMatchedRecordId(recordId);
          setSelectedId(recordId);
        }}
        onNoMatch={() => {
          setComparisonMode('custom-unmatched');
          setMatchedRecordId(null);
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

          if (card.key === 'rag') {
            return (
              <ModelResponseCard
                key={card.key}
                modelName={card.modelName}
                accessDescription={card.accessDescription}
                response={ragModelResponse}
                isLoading={isRagModelLoading}
                error={ragModelError}
                sources={ragSources}
              />
            );
          }

          if (!simulatedRecord) {
            return (
              <ModelResponseCard
                key={card.key}
                modelName={card.modelName}
                accessDescription={card.accessDescription}
                response={{
                  text: '',
                  grounding: 'High',
                  simulated: true,
                }}
                unavailableMessage={
                  showSimulatedUnavailable ? SIMULATED_UNAVAILABLE_MESSAGE : null
                }
              />
            );
          }

          return (
            <ModelResponseCard
              key={card.key}
              modelName={card.modelName}
              accessDescription={card.accessDescription}
              response={simulatedRecord[card.responseKey]}
            />
          );
        })}
      </section>

      <ComparisonExplanation notes={explanationNotes} />

      <div className="comparison-actions">
        <Link to={evaluateHref} className="button-link button-link--primary">
          Evaluate these responses
        </Link>
      </div>
    </>
  );
}
