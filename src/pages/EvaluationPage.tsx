import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import comparisonData from '../data/comparisonData.json';
import { ComparisonQuestionSelector } from '../components/ComparisonQuestionSelector';
import { FormFieldError } from '../components/FormFieldError';
import { ModelResponseCard } from '../components/ModelResponseCard';
import { PageHeader } from '../components/PageHeader';
import { useCourseId } from '../context/CourseContext';
import { useEvaluations } from '../hooks/useEvaluations';
import { coursePagePath } from '../lib/courseRoutes';
import type { ComparisonRecord, EvaluationRecord, ModelKey } from '../types';
import {
  formatEvaluationDate,
  generateEvaluationId,
  getModelLabel,
  getRecentEvaluations,
  MODEL_KEYS,
  resolveComparisonId,
} from '../utils/evaluationUtils';

const records = comparisonData as ComparisonRecord[];

const MODEL_CARDS = [
  {
    key: 'base' as const,
    modelName: 'Base Model',
    accessDescription: 'No syllabus context and no student-created fine-tuning examples.',
    responseKey: 'baseResponse' as const,
  },
  {
    key: 'rag' as const,
    modelName: 'RAG',
    accessDescription: 'Receives relevant syllabus passages at answer time.',
    responseKey: 'ragResponse' as const,
  },
  {
    key: 'fineTuned' as const,
    modelName: 'Fine-Tuned Model',
    accessDescription: 'Uses response behavior learned from prototype seed examples.',
    responseKey: 'fineTunedResponse' as const,
  },
  {
    key: 'fineTunedRag' as const,
    modelName: 'Fine-Tuned + RAG',
    accessDescription:
      'Uses both learned response behavior and retrieved syllabus context.',
    responseKey: 'fineTunedRagResponse' as const,
  },
];

const RATING_FIELDS = [
  { key: 'mostAccurate' as const, legend: 'Most accurate response' },
  { key: 'mostHelpful' as const, legend: 'Most helpful response' },
  { key: 'mostConcise' as const, legend: 'Most concise response' },
  { key: 'bestGrounded' as const, legend: 'Best grounded response' },
  { key: 'preferredModel' as const, legend: 'Overall preferred response' },
];

const COMMENT_MAX_LENGTH = 1000;

interface FormValues {
  mostAccurate: ModelKey | '';
  mostHelpful: ModelKey | '';
  mostConcise: ModelKey | '';
  bestGrounded: ModelKey | '';
  preferredModel: ModelKey | '';
  hallucinationFlags: ModelKey[];
  comment: string;
}

const INITIAL_VALUES: FormValues = {
  mostAccurate: '',
  mostHelpful: '',
  mostConcise: '',
  bestGrounded: '',
  preferredModel: '',
  hallucinationFlags: [],
  comment: '',
};

type RatingFieldKey = (typeof RATING_FIELDS)[number]['key'];

interface FormErrors {
  mostAccurate?: string;
  mostHelpful?: string;
  mostConcise?: string;
  bestGrounded?: string;
  preferredModel?: string;
  comment?: string;
}

export function EvaluationPage() {
  const courseId = useCourseId();
  const formId = useId();
  const [searchParams, setSearchParams] = useSearchParams();
  const {
    evaluations,
    loading,
    error,
    saving,
    saveError,
    addEvaluation,
    clearSaveError,
  } = useEvaluations();

  const comparisonParam = searchParams.get('comparison');
  const selectedId = useMemo(
    () => resolveComparisonId(comparisonParam, records),
    [comparisonParam],
  );

  const selectedRecord = useMemo(
    () => records.find((record) => record.id === selectedId) ?? records[0],
    [selectedId],
  );

  const [values, setValues] = useState<FormValues>(INITIAL_VALUES);
  const [errors, setErrors] = useState<FormErrors>({});
  const [submitted, setSubmitted] = useState(false);
  const [submittedId, setSubmittedId] = useState('');
  const formRef = useRef<HTMLFormElement>(null);
  const firstFieldsetErrorRef = useRef<HTMLFieldSetElement>(null);
  const commentErrorRef = useRef<HTMLTextAreaElement>(null);

  const recentEvaluations = useMemo(
    () => getRecentEvaluations(evaluations, records, 5),
    [evaluations],
  );

  const handleSelectComparison = useCallback(
    (id: string) => {
      setSearchParams({ comparison: id }, { replace: true });
      setSubmitted(false);
      setSubmittedId('');
    },
    [setSearchParams],
  );

  function updateRating(field: RatingFieldKey, value: ModelKey) {
    setValues((current) => ({ ...current, [field]: value }));
    setErrors((current) => ({ ...current, [field]: undefined }));
    setSubmitted(false);
    clearSaveError();
  }

  function toggleHallucinationFlag(model: ModelKey) {
    setValues((current) => {
      const flags = current.hallucinationFlags.includes(model)
        ? current.hallucinationFlags.filter((flag) => flag !== model)
        : [...current.hallucinationFlags, model];
      return { ...current, hallucinationFlags: flags };
    });
    setSubmitted(false);
    clearSaveError();
  }

  function updateComment(comment: string) {
    setValues((current) => ({ ...current, comment }));
    setErrors((current) => ({ ...current, comment: undefined }));
    setSubmitted(false);
    clearSaveError();
  }

  function validate(): FormErrors {
    const nextErrors: FormErrors = {};

    for (const field of RATING_FIELDS) {
      if (!values[field.key]) {
        nextErrors[field.key] = 'Select one model for this criterion.';
      }
    }

    if (values.comment.length > COMMENT_MAX_LENGTH) {
      nextErrors.comment = `Notes must be ${COMMENT_MAX_LENGTH} characters or fewer.`;
    }

    return nextErrors;
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    const nextErrors = validate();
    setErrors(nextErrors);

    if (Object.keys(nextErrors).length > 0) {
      setSubmitted(false);
      return;
    }

    if (!selectedRecord) {
      return;
    }

    const evaluation: EvaluationRecord = {
      id: generateEvaluationId(),
      comparisonId: selectedRecord.id,
      mostAccurate: values.mostAccurate as ModelKey,
      mostHelpful: values.mostHelpful as ModelKey,
      mostConcise: values.mostConcise as ModelKey,
      bestGrounded: values.bestGrounded as ModelKey,
      preferredModel: values.preferredModel as ModelKey,
      hallucinationFlags: values.hallucinationFlags,
      comment: values.comment.trim() || undefined,
      createdAt: new Date().toISOString(),
    };

    clearSaveError();

    try {
      const savedEvaluation = await addEvaluation(evaluation);
      setSubmitted(true);
      setSubmittedId(savedEvaluation.id);
    } catch {
      setSubmitted(false);
      setSubmittedId('');
    }
  }

  function handleEvaluateAnother() {
    setValues(INITIAL_VALUES);
    setErrors({});
    setSubmitted(false);
    setSubmittedId('');
    clearSaveError();
    formRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  useEffect(() => {
    if (Object.keys(errors).length === 0) {
      return;
    }
    if (errors.comment && commentErrorRef.current) {
      commentErrorRef.current.focus();
      return;
    }
    if (firstFieldsetErrorRef.current) {
      firstFieldsetErrorRef.current.focus();
    }
  }, [errors]);

  if (!selectedRecord) {
    return null;
  }

  const errorIds: Record<RatingFieldKey, string> = {
    mostAccurate: `${formId}-most-accurate-error`,
    mostHelpful: `${formId}-most-helpful-error`,
    mostConcise: `${formId}-most-concise-error`,
    bestGrounded: `${formId}-best-grounded-error`,
    preferredModel: `${formId}-preferred-model-error`,
  };

  const firstRatingErrorKey = RATING_FIELDS.find((field) => errors[field.key])?.key;

  return (
    <>
      <PageHeader
        title="Evaluation"
        description="Rate the model responses shown on the Model Comparison page. Your ratings help compare how different approaches perform on syllabus questions."
      />

      <aside className="evaluation-notice" aria-label="Evaluation notices">
        <p>
          <strong>Shared dataset:</strong> Evaluations are stored in Firebase Realtime Database
          and shared across browsers and devices.
        </p>
        <p>
          <strong>Response sources:</strong> On the Model Comparison page, Base Model, RAG,
          Fine-Tuned, and Fine-Tuned + RAG answers are live from the backend.
        </p>
      </aside>

      {error && (
        <p className="evaluation-status evaluation-status--error" role="alert">
          Could not load shared evaluations: {error}
        </p>
      )}

      {loading && (
        <p className="evaluation-status" role="status" aria-live="polite">
          Loading shared evaluations…
        </p>
      )}

      <ComparisonQuestionSelector
        records={records}
        selectedId={selectedRecord.id}
        onSelect={handleSelectComparison}
      />

      <section className="comparison-grid" aria-label="Model responses to evaluate">
        {MODEL_CARDS.map((card) => (
          <ModelResponseCard
            key={card.key}
            modelName={card.modelName}
            accessDescription={card.accessDescription}
            response={selectedRecord[card.responseKey]}
          />
        ))}
      </section>

      <section className="evaluation-form-section" aria-labelledby="evaluation-form-title">
        <h2 id="evaluation-form-title" className="evaluation-form-section__title">
          Rate these responses
        </h2>

        {submitted && (
          <div className="evaluation-success" role="status" aria-live="polite">
            <p>
              <strong>Evaluation saved.</strong> Your ratings for this question were saved to the
              shared dataset (ID: {submittedId}).
            </p>
            <div className="evaluation-success__actions">
              <button
                type="button"
                className="evaluation-form__submit"
                onClick={handleEvaluateAnother}
              >
                Evaluate another question
              </button>
              <Link to={coursePagePath(courseId, 'results')} className="button-link button-link--secondary">
                View results
              </Link>
            </div>
          </div>
        )}

        {saveError && (
          <p className="evaluation-status evaluation-status--error" role="alert">
            {saveError}
          </p>
        )}

        <form
          ref={formRef}
          className="evaluation-form"
          onSubmit={handleSubmit}
          noValidate
        >
          {RATING_FIELDS.map((field) => {
            const errorId = errorIds[field.key];
            const hasError = Boolean(errors[field.key]);
            const isFirstError = field.key === firstRatingErrorKey;
            return (
              <fieldset
                key={field.key}
                className="evaluation-form__fieldset"
                ref={isFirstError ? firstFieldsetErrorRef : undefined}
                tabIndex={isFirstError ? -1 : undefined}
              >
                <legend className="evaluation-form__legend">
                  {field.legend} <span className="evaluation-form__required">(required)</span>
                </legend>
                <div className="evaluation-form__radio-group" role="radiogroup">
                  {MODEL_KEYS.map((modelKey) => {
                    const inputId = `${formId}-${field.key}-${modelKey}`;
                    return (
                      <label key={modelKey} htmlFor={inputId} className="evaluation-form__radio-label">
                        <input
                          type="radio"
                          id={inputId}
                          name={field.key}
                          value={modelKey}
                          checked={values[field.key] === modelKey}
                          onChange={() => updateRating(field.key, modelKey)}
                          aria-invalid={hasError}
                          aria-describedby={hasError ? errorId : undefined}
                          disabled={saving}
                        />
                        {getModelLabel(modelKey)}
                      </label>
                    );
                  })}
                </div>
                {hasError && (
                  <FormFieldError id={errorId} message={errors[field.key]!} />
                )}
              </fieldset>
            );
          })}

          <fieldset className="evaluation-form__fieldset">
            <legend className="evaluation-form__legend">
              Which responses contain unsupported or invented information?
            </legend>
            <p className="evaluation-form__hint">
              Select none if you do not believe any response contains unsupported information.
            </p>
            <div className="evaluation-form__checkbox-group">
              {MODEL_KEYS.map((modelKey) => {
                const inputId = `${formId}-hallucination-${modelKey}`;
                return (
                  <label key={modelKey} htmlFor={inputId} className="evaluation-form__checkbox-label">
                    <input
                      type="checkbox"
                      id={inputId}
                      checked={values.hallucinationFlags.includes(modelKey)}
                      onChange={() => toggleHallucinationFlag(modelKey)}
                      disabled={saving}
                    />
                    {getModelLabel(modelKey)}
                  </label>
                );
              })}
            </div>
          </fieldset>

          <div className="evaluation-form__field">
            <label htmlFor={`${formId}-comment`} className="evaluation-form__label">
              Optional evaluation notes
            </label>
            <textarea
              id={`${formId}-comment`}
              className={`evaluation-form__textarea${errors.comment ? ' evaluation-form__input--error' : ''}`}
              value={values.comment}
              onChange={(event) => updateComment(event.target.value)}
              rows={4}
              maxLength={COMMENT_MAX_LENGTH}
              aria-invalid={Boolean(errors.comment)}
              aria-describedby={errors.comment ? `${formId}-comment-error` : `${formId}-comment-hint`}
              ref={commentErrorRef}
              disabled={saving}
            />
            <p id={`${formId}-comment-hint`} className="evaluation-form__char-count">
              {values.comment.length} / {COMMENT_MAX_LENGTH} characters
            </p>
            {errors.comment && (
              <FormFieldError id={`${formId}-comment-error`} message={errors.comment} />
            )}
          </div>

          <button type="submit" className="evaluation-form__submit" disabled={saving}>
            {saving ? 'Saving evaluation…' : 'Submit evaluation'}
          </button>
        </form>
      </section>

      {!loading && recentEvaluations.length > 0 && (
        <section className="evaluation-recent" aria-labelledby="evaluation-recent-title">
          <h2 id="evaluation-recent-title" className="evaluation-recent__title">
            Recent evaluations
          </h2>
          <ul className="evaluation-recent__list">
            {recentEvaluations.map((item) => (
              <li key={item.id} className="evaluation-recent__item">
                <p className="evaluation-recent__question">{item.question}</p>
                <dl className="evaluation-recent__meta">
                  <div className="evaluation-recent__meta-row">
                    <dt>Preferred</dt>
                    <dd>{item.preferredModel}</dd>
                  </div>
                  <div className="evaluation-recent__meta-row">
                    <dt>Most accurate</dt>
                    <dd>{item.mostAccurate}</dd>
                  </div>
                  <div className="evaluation-recent__meta-row">
                    <dt>Submitted</dt>
                    <dd>{formatEvaluationDate(item.createdAt)}</dd>
                  </div>
                </dl>
              </li>
            ))}
          </ul>
        </section>
      )}
    </>
  );
}
