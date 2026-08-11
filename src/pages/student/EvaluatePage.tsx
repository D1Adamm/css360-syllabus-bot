import { useEffect, useId, useRef, useState } from 'react';
import { APPROACHES } from '../../components/compare/approaches';
import { CriterionRow } from '../../components/evaluate/CriterionRow';
import { ResponseStrip } from '../../components/evaluate/ResponseStrip';
import { Button, LinkButton } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { EmptyState } from '../../components/ui/EmptyState';
import { PageHeader } from '../../components/ui/PageHeader';
import { useCourseId } from '../../context/CourseContext';
import { useComparisonRunStore } from '../../context/ComparisonRunContext';
import { useEvaluations } from '../../hooks/useEvaluations';
import { toUserMessage } from '../../lib/errorMessages';
import { studentCoursePath } from '../../lib/roleRoutes';
import type { EvaluationRecord, ModelKey } from '../../types';
import { generateEvaluationId } from '../../utils/evaluationUtils';

const CRITERIA = [
  { key: 'mostAccurate' as const, legend: 'Which answer was most accurate?' },
  { key: 'mostHelpful' as const, legend: 'Which was most helpful?' },
  { key: 'mostConcise' as const, legend: 'Which was most concise?' },
  { key: 'bestGrounded' as const, legend: 'Which stayed closest to the syllabus?' },
  { key: 'preferredModel' as const, legend: 'Which did you prefer overall?' },
];

type CriterionKey = (typeof CRITERIA)[number]['key'];

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

type FormErrors = Partial<Record<CriterionKey | 'comment', string>>;

/**
 * Rates the four responses the student just saw.
 *
 * The run comes from `ComparisonRunContext`, not from the bundled example
 * data, so what is rated is exactly what Compare generated. When the question
 * matched a predefined example the rating keeps that id — which is what
 * results aggregation has always grouped on, so older records are unaffected.
 * Free-text questions store a synthetic id plus the wording, and aggregation
 * falls back to that wording.
 */
export function EvaluatePage() {
  const courseId = useCourseId();
  const formId = useId();
  const { getRun, clearRun } = useComparisonRunStore();
  const { saving, saveError, addEvaluation, clearSaveError } = useEvaluations();

  const run = getRun(courseId);

  const [values, setValues] = useState<FormValues>(INITIAL_VALUES);
  const [errors, setErrors] = useState<FormErrors>({});
  const [submitted, setSubmitted] = useState(false);
  const firstErrorRef = useRef<HTMLFieldSetElement>(null);

  useEffect(() => {
    if (Object.keys(errors).length > 0) {
      firstErrorRef.current?.focus();
    }
  }, [errors]);

  function updateCriterion(field: CriterionKey, value: ModelKey) {
    setValues((current) => ({ ...current, [field]: value }));
    setErrors((current) => ({ ...current, [field]: undefined }));
    setSubmitted(false);
    clearSaveError();
  }

  function toggleFlag(model: ModelKey) {
    setValues((current) => ({
      ...current,
      hallucinationFlags: current.hallucinationFlags.includes(model)
        ? current.hallucinationFlags.filter((flag) => flag !== model)
        : [...current.hallucinationFlags, model],
    }));
    setSubmitted(false);
    clearSaveError();
  }

  function validate(): FormErrors {
    const next: FormErrors = {};
    for (const criterion of CRITERIA) {
      if (!values[criterion.key]) {
        next[criterion.key] = 'Choose one answer.';
      }
    }
    if (values.comment.length > COMMENT_MAX_LENGTH) {
      next.comment = `Notes must be ${COMMENT_MAX_LENGTH} characters or fewer.`;
    }
    return next;
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();

    if (!run) {
      return;
    }

    const nextErrors = validate();
    setErrors(nextErrors);

    if (Object.keys(nextErrors).length > 0) {
      setSubmitted(false);
      return;
    }

    const evaluation: EvaluationRecord = {
      id: generateEvaluationId(),
      comparisonId: run.matchedComparisonId ?? `question-${run.runId}`,
      mostAccurate: values.mostAccurate as ModelKey,
      mostHelpful: values.mostHelpful as ModelKey,
      mostConcise: values.mostConcise as ModelKey,
      bestGrounded: values.bestGrounded as ModelKey,
      preferredModel: values.preferredModel as ModelKey,
      hallucinationFlags: values.hallucinationFlags,
      comment: values.comment.trim() || undefined,
      createdAt: new Date().toISOString(),
      runId: run.runId,
      questionText: run.question,
      courseId: run.courseId,
    };

    clearSaveError();

    try {
      await addEvaluation(evaluation);
      setSubmitted(true);
    } catch {
      setSubmitted(false);
    }
  }

  if (!run) {
    return (
      <div className="ui-stack ui-stack--loose">
        <PageHeader
          title="Evaluate Responses"
          description="Rate the responses to a course question."
        />
        <EmptyState
          size="full"
          illustration="contribute"
          title="Nothing to evaluate yet"
          description="Ask a question on Compare first. The four answers you get there are what you'll rate here."
          action={
            <LinkButton
              to={studentCoursePath(courseId, 'compare')}
              variant="primary"
              iconRight="forward"
            >
              Go to Compare
            </LinkButton>
          }
        />
      </div>
    );
  }

  if (submitted) {
    return (
      <div className="ui-stack ui-stack--loose">
        <PageHeader
          title="Evaluate Responses"
          description="Rate the responses to a course question."
        />
        <EmptyState
          size="full"
          illustration="model-ready"
          title="Thanks — your ratings were recorded"
          description="Your feedback helps show which approach answers course questions best."
          action={
            <>
              <LinkButton
                to={studentCoursePath(courseId, 'compare')}
                variant="primary"
                onClick={() => clearRun(courseId)}
              >
                Compare another question
              </LinkButton>
              <LinkButton
                to={studentCoursePath(courseId, 'contribute')}
                variant="tertiary"
              >
                Contribute a question
              </LinkButton>
            </>
          }
        />
      </div>
    );
  }

  // An approach that could not answer cannot win a criterion.
  const unavailable = APPROACHES.filter((approach) =>
    Boolean(run.responses[approach.key]?.error),
  ).map((approach) => approach.key);

  const answeredCount = APPROACHES.length - unavailable.length;
  const firstErrorKey = CRITERIA.find((criterion) => errors[criterion.key])?.key;

  // With nothing to compare there is nothing to rate; offer the way out
  // rather than a form that cannot be completed.
  if (answeredCount === 0) {
    return (
      <div className="ui-stack ui-stack--loose">
        <PageHeader
          title="Evaluate Responses"
          description="Rate the responses to a course question."
        />
        <EmptyState
          size="full"
          illustration="contribute"
          title="No answers came through"
          description="None of the approaches could answer this question, so there is nothing to rate yet. Try asking again in a moment."
          action={
            <LinkButton
              to={studentCoursePath(courseId, 'compare')}
              variant="primary"
              iconRight="forward"
              onClick={() => clearRun(courseId)}
            >
              Ask again
            </LinkButton>
          }
        />
      </div>
    );
  }

  return (
    <div className="ui-stack ui-stack--section">
      <PageHeader
        title="Evaluate Responses"
        description="Rate the four answers you just compared."
      />

      {answeredCount < APPROACHES.length && (
        <Callout tone="info" title="Some answers are unavailable">
          You can still rate the answers that did come through.
        </Callout>
      )}

      <ResponseStrip run={run} />

      <form className="evaluate__form" onSubmit={handleSubmit} noValidate>
        {saveError && (
          <Callout tone="danger" title="Not saved">
            {
              toUserMessage(new Error(saveError), {
                audience: 'student',
                context: 'evaluation-save',
              }).message
            }
          </Callout>
        )}

        <div className="evaluate__criteria">
          {CRITERIA.map((criterion) => (
            <CriterionRow
              key={criterion.key}
              legend={criterion.legend}
              name={`${formId}-${criterion.key}`}
              value={values[criterion.key]}
              error={errors[criterion.key]}
              errorId={`${formId}-${criterion.key}-error`}
              disabled={saving}
              unavailable={unavailable}
              onChange={(value) => updateCriterion(criterion.key, value)}
              fieldsetRef={criterion.key === firstErrorKey ? firstErrorRef : undefined}
            />
          ))}
        </div>

        <fieldset className="evaluate__flags">
          <legend className="criterion__legend">
            Did any answer include something the syllabus doesn&apos;t support?
          </legend>
          <p className="criterion__hint">
            Leave these unchecked if nothing looked invented.
          </p>
          <div className="evaluate__flag-options">
            {APPROACHES.map((approach, index) => {
              const inputId = `${formId}-flag-${approach.key}`;
              return (
                <label
                  key={approach.key}
                  htmlFor={inputId}
                  className={`evaluate__flag${
                    values.hallucinationFlags.includes(approach.key)
                      ? ' evaluate__flag--selected'
                      : ''
                  }`}
                >
                  <input
                    type="checkbox"
                    id={inputId}
                    checked={values.hallucinationFlags.includes(approach.key)}
                    onChange={() => toggleFlag(approach.key)}
                    disabled={saving || unavailable.includes(approach.key)}
                  />
                  <span className="criterion__marker" aria-hidden="true">
                    {String.fromCharCode(65 + index)}
                  </span>
                  {approach.label}
                </label>
              );
            })}
          </div>
        </fieldset>

        <div className="evaluate__notes">
          <label className="ui-field__label" htmlFor={`${formId}-comment`}>
            Anything else? <span className="ui-field__requirement">(optional)</span>
          </label>
          <textarea
            id={`${formId}-comment`}
            className="evaluate__textarea"
            value={values.comment}
            onChange={(event) => {
              const comment = event.target.value;
              setValues((current) => ({ ...current, comment }));
              setErrors((current) => ({ ...current, comment: undefined }));
            }}
            rows={3}
            maxLength={COMMENT_MAX_LENGTH}
            disabled={saving}
            placeholder="What made one answer better than the others?"
          />
          <p className="ui-text-xs ui-text-muted">
            {values.comment.length} / {COMMENT_MAX_LENGTH}
          </p>
          {errors.comment && <p className="ui-field__error">{errors.comment}</p>}
        </div>

        <div className="evaluate__submit">
          <Button type="submit" variant="primary" loading={saving} loadingLabel="Saving…">
            Submit evaluation
          </Button>
        </div>
      </form>
    </div>
  );
}
