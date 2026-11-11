import { useId, useMemo, useState } from 'react';
import { Button, LinkButton } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { EmptyState } from '../../components/ui/EmptyState';
import { ErrorState } from '../../components/ui/ErrorState';
import { PageHeader } from '../../components/ui/PageHeader';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { StatusPill } from '../../components/ui/StatusPill';
import { useCourseId } from '../../context/CourseContext';
import { useClassroomPreparation } from '../../hooks/useClassroomPreparation';
import {
  ANSWER_STATE_LABELS,
  APPROACH_KEYS,
  APPROACH_SHORT_LABELS,
  SOURCE_FILTERS,
  filterQuestions,
  preparationPresentation,
  primaryAction,
  type AnswerState,
  type PreparationQuestion,
  type PreparationStatus,
} from '../../lib/classroomPreparation';
import { adminCoursePath, professorCourseHomePath } from '../../lib/roleRoutes';

export interface ClassroomPreparationPageProps {
  /** Which interface the page is mounted in. The page itself is the same. */
  audience?: 'professor' | 'admin';
}

/** The mark beside each indicator's label, so its state does not rest on colour alone. */
const ANSWER_STATE_MARKS: Record<AnswerState, string> = {
  ready: '✓',
  missing: '○',
  outdated: '↻',
  unavailable: '!',
  not_expected: '–',
};

const INELIGIBLE_NOTES: Record<string, string> = {
  rejected: 'Rejected in review, so it is no longer prepared.',
  student_unapproved: "A student's question that is not approved, so it is no longer prepared.",
  empty: 'This question has no text, so it cannot be prepared.',
};

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? '' : 's'}`;
}

function AnswerIndicators({
  question,
  preparation,
}: {
  question: PreparationQuestion;
  preparation: PreparationStatus;
}) {
  if (!question.answers) {
    return null;
  }
  return (
    <ul className="prep-answers" aria-label={`Answers for: ${question.question}`}>
      {APPROACH_KEYS.map((key) => {
        const state = question.answers?.[key] ?? 'missing';
        const label = preparation.conditions[key]?.label ?? APPROACH_SHORT_LABELS[key];
        return (
          <li
            key={key}
            className={`prep-answer prep-answer--${state}`}
            title={`${label}: ${ANSWER_STATE_LABELS[state]}`}
          >
            <span aria-hidden="true" className="prep-answer__mark">
              {ANSWER_STATE_MARKS[state]}
            </span>
            <span aria-hidden="true">{APPROACH_SHORT_LABELS[key]}</span>
            <span className="ui-visually-hidden">
              {label}: {ANSWER_STATE_LABELS[state]}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

/**
 * Classroom answers for one course: choose questions, prepare their answers.
 *
 * Staff pick some of the course's existing questions; the server then asks
 * each of the four approaches for its real answer, one at a time, and keeps
 * every answer as soon as it exists. Students asking one of those questions
 * get the kept answers at once. Preparation can be stopped at any moment
 * without losing anything, and carried on later.
 *
 * Everything shown is read from the server. A run belongs to the server, not
 * to this page, so leaving, refreshing or coming back later shows the same
 * thing. The one page serves professors and administrators; `audience` only
 * changes where "back" goes.
 */
export function ClassroomPreparationPage({
  audience = 'professor',
}: ClassroomPreparationPageProps = {}) {
  const courseId = useCourseId();
  const asAdmin = audience === 'admin';
  const { state, busy, actionError, retry, start, stop, saveSelection } =
    useClassroomPreparation(courseId);

  const fieldId = useId();
  const [search, setSearch] = useState('');
  const [source, setSource] = useState('all');
  const [selectedOnly, setSelectedOnly] = useState(false);

  const preparation = state.status === 'ready' ? state.preparation : null;
  const questions = useMemo(() => preparation?.questions ?? [], [preparation]);
  const visible = useMemo(
    () => filterQuestions(questions, { search, source, selectedOnly }),
    [questions, search, source, selectedOnly],
  );
  const selectedIds = useMemo(
    () => questions.filter((question) => question.selected).map((question) => question.seedId),
    [questions],
  );

  const backTo = asAdmin ? adminCoursePath(courseId) : professorCourseHomePath(courseId);

  const header = (
    <PageHeader
      title="Classroom answers"
      eyebrow={asAdmin ? 'Admin' : undefined}
      description="Prepare the four AI answers to chosen course questions before class, so students who ask those questions get them without waiting."
      actions={
        <LinkButton to={backTo} variant="tertiary" iconLeft="previous">
          Back to course
        </LinkButton>
      }
    />
  );

  if (state.status === 'loading') {
    return (
      <div className="ui-stack ui-stack--loose">
        {header}
        <p className="ui-text-muted" role="status">
          Loading classroom answers…
        </p>
      </div>
    );
  }

  if (!preparation) {
    return (
      <div className="ui-stack ui-stack--loose">
        {header}
        <ErrorState
          title="Classroom answers unavailable"
          message={
            state.status === 'unavailable'
              ? `${state.message} Answers that were already prepared are not affected.`
              : 'Classroom answers could not be loaded.'
          }
          onRetry={retry}
        />
      </div>
    );
  }

  const { counts } = preparation;
  const presentation = preparationPresentation(preparation.state);
  const action = primaryAction(preparation);
  const percent =
    counts.expectedAnswers > 0
      ? Math.round((counts.readyAnswers / counts.expectedAnswers) * 100)
      : 0;

  const toggle = (seedId: string, checked: boolean) => {
    const next = new Set(selectedIds);
    if (checked) {
      next.add(seedId);
    } else {
      next.delete(seedId);
    }
    saveSelection(questions.filter((q) => next.has(q.seedId)).map((q) => q.seedId));
  };

  const selectableVisible = visible.filter((question) => question.eligible);
  const allVisibleSelected =
    selectableVisible.length > 0 && selectableVisible.every((question) => question.selected);
  const anyVisibleSelected = visible.some((question) => question.selected);

  const setVisible = (checked: boolean) => {
    const next = new Set(selectedIds);
    for (const question of checked ? selectableVisible : visible) {
      if (checked) {
        next.add(question.seedId);
      } else {
        next.delete(question.seedId);
      }
    }
    saveSelection(questions.filter((q) => next.has(q.seedId)).map((q) => q.seedId));
  };

  return (
    <div className="ui-stack ui-stack--loose">
      {header}

      <section className="prep-summary" aria-labelledby={`${fieldId}-summary`}>
        <div className="prep-summary__head">
          <h2 className="prep-summary__title" id={`${fieldId}-summary`}>
            Preparation
          </h2>
          <StatusPill tone={presentation.tone}>{presentation.label}</StatusPill>
        </div>

        <p className="prep-summary__message" role="status" aria-live="polite">
          {preparation.message}
        </p>

        {counts.selectedQuestions > 0 && (
          <>
            <dl className="prep-stats">
              <div className="prep-stat">
                <dt>Answers ready</dt>
                <dd>
                  <strong>{counts.readyAnswers}</strong> of {counts.expectedAnswers}
                </dd>
              </div>
              <div className="prep-stat">
                <dt>Questions fully prepared</dt>
                <dd>
                  <strong>{counts.fullyPreparedQuestions}</strong> of {counts.selectedQuestions}
                </dd>
              </div>
              <div className="prep-stat">
                <dt>Still to prepare</dt>
                <dd>
                  <strong>{counts.missingAnswers}</strong>
                  {counts.outdatedAnswers > 0 && (
                    <span className="prep-stat__note"> ({counts.outdatedAnswers} out of date)</span>
                  )}
                </dd>
              </div>
            </dl>

            <div
              className="prep-progress"
              role="progressbar"
              aria-label="Answers ready"
              aria-valuemin={0}
              aria-valuemax={counts.expectedAnswers}
              aria-valuenow={counts.readyAnswers}
              aria-valuetext={`${counts.readyAnswers} of ${plural(counts.expectedAnswers, 'answer')} ready`}
            >
              <span className="prep-progress__bar" style={{ width: `${percent}%` }} />
            </div>
          </>
        )}

        {action && (
          <div className="prep-summary__actions">
            {action.kind === 'start' && (
              <Button variant="primary" onClick={start} disabled={busy}>
                {action.label}
              </Button>
            )}
            {action.kind === 'stop' && (
              <Button variant="secondary" onClick={stop} disabled={busy}>
                {action.label}
              </Button>
            )}
            {action.kind === 'stopping' && (
              <Button variant="secondary" disabled>
                {action.label}
              </Button>
            )}
            {action.kind === 'stop' && (
              <p className="prep-summary__hint">
                Stopping keeps every answer prepared so far. You can carry on later.
              </p>
            )}
          </div>
        )}

        {actionError && (
          <Callout tone="danger" title="That did not work" live>
            {actionError}
          </Callout>
        )}

        {preparation.notices.map((notice) => (
          <Callout key={notice} tone="info">
            {notice}
          </Callout>
        ))}
      </section>

      <section className="ui-stack ui-stack--snug" aria-labelledby={`${fieldId}-questions`}>
        <SectionHeader
          id={`${fieldId}-questions`}
          title="Questions"
          description="Choose which of this course's existing questions to prepare. Choosing a question here does not suggest it to students and does not approve it for training; only the question is used, never its example answer."
          divider
        />

        {questions.length === 0 ? (
          <EmptyState
            illustration="empty-course"
            title="No questions to choose from yet"
            description="This course has no example questions yet. Add or generate examples for the course first, then come back to choose which ones to prepare."
          />
        ) : (
          <>
            <div className="review-filters">
              <div className="ui-field review-filters__search">
                <label className="ui-field__label" htmlFor={`${fieldId}-search`}>
                  Search questions
                </label>
                <div className="ui-field__control">
                  <input
                    id={`${fieldId}-search`}
                    type="search"
                    value={search}
                    placeholder="Question text…"
                    onChange={(event) => setSearch(event.target.value)}
                  />
                </div>
              </div>

              <div className="ui-field">
                <label className="ui-field__label" htmlFor={`${fieldId}-source`}>
                  Source
                </label>
                <div className="ui-field__control">
                  <select
                    id={`${fieldId}-source`}
                    value={source}
                    onChange={(event) => setSource(event.target.value)}
                  >
                    {SOURCE_FILTERS.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="review-filters__toggle">
                <input
                  type="checkbox"
                  id={`${fieldId}-selected-only`}
                  checked={selectedOnly}
                  onChange={(event) => setSelectedOnly(event.target.checked)}
                />
                <label htmlFor={`${fieldId}-selected-only`}>Selected only</label>
              </div>
            </div>

            <div className="review-list-bar">
              <p className="review-list-bar__count" aria-live="polite">
                {visible.length} of {plural(questions.length, 'question')} shown ·{' '}
                {selectedIds.length} selected
              </p>
              <div className="review-list-bar__controls">
                <Button
                  variant="tertiary"
                  size="sm"
                  onClick={() => setVisible(true)}
                  disabled={selectableVisible.length === 0 || allVisibleSelected}
                >
                  Select all shown
                </Button>
                <Button
                  variant="tertiary"
                  size="sm"
                  onClick={() => setVisible(false)}
                  disabled={!anyVisibleSelected}
                >
                  Clear shown
                </Button>
              </div>
            </div>

            <p className="prep-legend">
              For each selected question: <span className="prep-legend__item">✓ ready</span>
              <span className="prep-legend__item">○ not prepared yet</span>
              <span className="prep-legend__item">↻ out of date</span>
              <span className="prep-legend__item">! cannot be prepared right now</span>
              <span className="prep-legend__item">– not used by this course</span>
            </p>

            {visible.length === 0 ? (
              <EmptyState
                illustration="empty-course"
                size="compact"
                title="No matching questions"
                description="Try a different search or source."
              />
            ) : (
              <ul className="prep-list" aria-label="Course questions">
                {visible.map((question) => {
                  const inputId = `${fieldId}-q-${question.seedId}`;
                  return (
                    <li
                      key={question.seedId}
                      className={`prep-row${question.selected ? ' prep-row--selected' : ''}`}
                    >
                      <input
                        type="checkbox"
                        id={inputId}
                        className="prep-row__check"
                        checked={question.selected}
                        // A question that stopped being eligible can only be taken out.
                        disabled={!question.eligible && !question.selected}
                        onChange={(event) => toggle(question.seedId, event.target.checked)}
                      />
                      <div className="prep-row__body">
                        <label htmlFor={inputId} className="prep-row__question">
                          {question.question}
                        </label>
                        <p className="prep-row__meta">
                          {question.sourceLabel}
                          {!question.eligible && (
                            <span className="prep-row__warning">
                              {' · '}
                              {INELIGIBLE_NOTES[question.ineligibleReason ?? ''] ??
                                'This question can no longer be prepared.'}
                            </span>
                          )}
                        </p>
                      </div>
                      <AnswerIndicators question={question} preparation={preparation} />
                    </li>
                  );
                })}
              </ul>
            )}
          </>
        )}
      </section>
    </div>
  );
}
