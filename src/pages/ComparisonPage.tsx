import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import comparisonData from '../data/comparisonData.json';
import { ComparisonExplanation } from '../components/ComparisonExplanation';
import { ComparisonQuestionSelector } from '../components/ComparisonQuestionSelector';
import { CustomQuestionMatcher } from '../components/CustomQuestionMatcher';
import { ModelResponseCard } from '../components/ModelResponseCard';
import { PageHeader } from '../components/PageHeader';
import { useCourseId } from '../context/CourseContext';
import {
  ApiError,
  generateBaseModel,
  generateFineTuned,
  generateFineTunedRag,
  generateRag,
} from '../lib/api';
import { coursePagePath } from '../lib/courseRoutes';
import type { ComparisonRecord, ComparisonResponse } from '../types';
import {
  type ComparisonMode,
  resolveSimulatedRecord,
} from '../utils/comparisonPageState';

const records = comparisonData as ComparisonRecord[];

const MODEL_CARDS = [
  {
    key: 'base',
    modelName: 'Base Model',
    accessDescription: 'No syllabus context and no student-created fine-tuning examples.',
  },
  {
    key: 'rag',
    modelName: 'RAG',
    accessDescription: 'Receives relevant syllabus passages at answer time.',
  },
  {
    key: 'fineTuned',
    modelName: 'Fine-Tuned Model',
    accessDescription: 'Uses response behavior learned from prototype seed examples.',
  },
  {
    key: 'fineTunedRag',
    modelName: 'Fine-Tuned + RAG',
    accessDescription:
      'Uses both learned response behavior and retrieved syllabus context.',
  },
] as const;

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

const DEFAULT_FINE_TUNED_RESPONSE: ComparisonResponse = {
  text: '',
  grounding: 'Medium',
  simulated: false,
};

const DEFAULT_FINE_TUNED_RAG_RESPONSE: ComparisonResponse = {
  text: '',
  grounding: 'High',
  simulated: false,
};

type LiveModelState =
  | { status: 'idle' }
  | { status: 'loading'; question: string }
  | { status: 'success'; question: string; response: ComparisonResponse }
  | { status: 'error'; question: string; message: string };

type GroundedModelState =
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
  const courseId = useCourseId();
  const [selectedId, setSelectedId] = useState(records[0]?.id ?? '');
  const [activeQuestion, setActiveQuestion] = useState(records[0]?.question ?? '');
  const [comparisonMode, setComparisonMode] = useState<ComparisonMode>('predefined');
  const [matchedRecordId, setMatchedRecordId] = useState<string | null>(null);
  const [baseModelState, setBaseModelState] = useState<LiveModelState>({ status: 'idle' });
  const [ragModelState, setRagModelState] = useState<GroundedModelState>({ status: 'idle' });
  const [fineTunedModelState, setFineTunedModelState] = useState<LiveModelState>({
    status: 'idle',
  });
  const [fineTunedRagModelState, setFineTunedRagModelState] = useState<GroundedModelState>({
    status: 'idle',
  });
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

  const notesRecord = useMemo(() => {
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
      const result = await generateBaseModel(courseId, trimmedQuestion);

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
  }, [courseId]);

  const loadRagModelResponse = useCallback(async (question: string) => {
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion) {
      return;
    }

    setRagModelState({ status: 'loading', question: trimmedQuestion });

    try {
      const result = await generateRag(courseId, trimmedQuestion);

      setRagModelState({
        status: 'success',
        question: trimmedQuestion,
        response: {
          text: result.answer,
          grounding: 'High',
          simulated: false,
        },
        sources: result.sources.map((source) => source.sectionTitle),
      });
    } catch (error) {
      setRagModelState({
        status: 'error',
        question: trimmedQuestion,
        message: getApiErrorMessage(error, 'The RAG response could not be loaded.'),
      });
    }
  }, [courseId]);

  const loadFineTunedModelResponse = useCallback(async (question: string) => {
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion) {
      return;
    }

    setFineTunedModelState({ status: 'loading', question: trimmedQuestion });

    try {
      const result = await generateFineTuned(courseId, trimmedQuestion);

      setFineTunedModelState({
        status: 'success',
        question: trimmedQuestion,
        response: {
          text: result.answer,
          grounding: 'Medium',
          simulated: false,
        },
      });
    } catch (error) {
      setFineTunedModelState({
        status: 'error',
        question: trimmedQuestion,
        message: getApiErrorMessage(
          error,
          'The Fine-Tuned Model response could not be loaded.',
        ),
      });
    }
  }, [courseId]);

  const loadFineTunedRagModelResponse = useCallback(async (question: string) => {
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion) {
      return;
    }

    setFineTunedRagModelState({ status: 'loading', question: trimmedQuestion });

    try {
      const result = await generateFineTunedRag(courseId, trimmedQuestion);

      setFineTunedRagModelState({
        status: 'success',
        question: trimmedQuestion,
        response: {
          text: result.answer,
          grounding: 'High',
          simulated: false,
        },
        sources: result.sources.map((source) => source.sectionTitle),
      });
    } catch (error) {
      setFineTunedRagModelState({
        status: 'error',
        question: trimmedQuestion,
        message: getApiErrorMessage(
          error,
          'The Fine-Tuned + RAG response could not be loaded.',
        ),
      });
    }
  }, [courseId]);

  useEffect(() => {
    void Promise.all([
      loadBaseModelResponse(activeQuestion),
      loadRagModelResponse(activeQuestion),
      loadFineTunedModelResponse(activeQuestion),
      loadFineTunedRagModelResponse(activeQuestion),
    ]);
  }, [
    activeQuestion,
    loadBaseModelResponse,
    loadFineTunedModelResponse,
    loadFineTunedRagModelResponse,
    loadRagModelResponse,
  ]);

  if (!selectedRecord) {
    return null;
  }

  const evaluateHref = `${coursePagePath(courseId, 'evaluate')}?comparison=${notesRecord?.id ?? selectedRecord.id}`;
  const isBaseModelLoading = baseModelState.status === 'loading';
  const isRagModelLoading = ragModelState.status === 'loading';
  const isFineTunedModelLoading = fineTunedModelState.status === 'loading';
  const isFineTunedRagModelLoading = fineTunedRagModelState.status === 'loading';
  const isLiveRequestInFlight =
    isBaseModelLoading ||
    isRagModelLoading ||
    isFineTunedModelLoading ||
    isFineTunedRagModelLoading;
  const baseModelError = baseModelState.status === 'error' ? baseModelState.message : null;
  const ragModelError = ragModelState.status === 'error' ? ragModelState.message : null;
  const fineTunedModelError =
    fineTunedModelState.status === 'error' ? fineTunedModelState.message : null;
  const fineTunedRagModelError =
    fineTunedRagModelState.status === 'error' ? fineTunedRagModelState.message : null;
  const baseModelResponse =
    baseModelState.status === 'success' ? baseModelState.response : DEFAULT_BASE_RESPONSE;
  const ragModelResponse =
    ragModelState.status === 'success' ? ragModelState.response : DEFAULT_RAG_RESPONSE;
  const fineTunedModelResponse =
    fineTunedModelState.status === 'success'
      ? fineTunedModelState.response
      : DEFAULT_FINE_TUNED_RESPONSE;
  const fineTunedRagModelResponse =
    fineTunedRagModelState.status === 'success'
      ? fineTunedRagModelState.response
      : DEFAULT_FINE_TUNED_RAG_RESPONSE;
  const ragSources = ragModelState.status === 'success' ? ragModelState.sources : [];
  const fineTunedRagSources =
    fineTunedRagModelState.status === 'success' ? fineTunedRagModelState.sources : [];
  const explanationNotes =
    comparisonMode === 'custom-unmatched'
      ? 'All four live models answered your custom question. Evaluation notes below still refer to the closest predefined comparison example when available.'
      : (notesRecord?.notes ?? selectedRecord.notes);

  return (
    <>
      <PageHeader
        title="Model Comparison"
        description="Compare how different model approaches may answer the same syllabus question. Each configuration has different access to syllabus content and seed-example training data."
      />

      <aside className="comparison-notice" aria-label="Live model notice">
        <p>
          <strong>Live comparison:</strong> Base Model, RAG, Fine-Tuned, and Fine-Tuned + RAG
          responses come from the FastAPI backend. Fine-Tuned paths require{' '}
          <code>FINETUNED_SERVICE_URL</code>; RAG paths require a course syllabus index.
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

          if (card.key === 'fineTuned') {
            return (
              <ModelResponseCard
                key={card.key}
                modelName={card.modelName}
                accessDescription={card.accessDescription}
                response={fineTunedModelResponse}
                isLoading={isFineTunedModelLoading}
                error={fineTunedModelError}
              />
            );
          }

          return (
            <ModelResponseCard
              key={card.key}
              modelName={card.modelName}
              accessDescription={card.accessDescription}
              response={fineTunedRagModelResponse}
              isLoading={isFineTunedRagModelLoading}
              error={fineTunedRagModelError}
              sources={fineTunedRagSources}
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
