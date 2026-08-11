import { useCallback } from 'react';
import { APPROACHES } from '../../components/compare/approaches';
import { ApproachCard } from '../../components/compare/ApproachCard';
import { QuestionAsk } from '../../components/compare/QuestionAsk';
import { LinkButton } from '../../components/ui/Button';
import { EmptyState } from '../../components/ui/EmptyState';
import { PageHeader } from '../../components/ui/PageHeader';
import { useCourseId } from '../../context/CourseContext';
import {
  useComparisonRunStore,
  type ComparisonRun,
} from '../../context/ComparisonRunContext';
import { useComparisonRun } from '../../hooks/useComparisonRun';
import { useQuestionSuggestions } from '../../hooks/useQuestionSuggestions';
import { studentCoursePath } from '../../lib/roleRoutes';

/**
 * Ask one question, compare four answers.
 *
 * Everything on this page is scoped to the course id in the URL: the four model
 * requests, the suggested questions, and the stored run. `CourseRoute` keys on
 * that id, so switching courses remounts this page rather than carrying one
 * course's answers into another.
 */
export function ComparePage() {
  const courseId = useCourseId();
  const { saveRun, getRun } = useComparisonRunStore();
  const suggestions = useQuestionSuggestions(courseId);

  const handleComplete = useCallback(
    (run: ComparisonRun) => {
      saveRun(run);
    },
    [saveRun],
  );

  const { states, activeQuestion, isRunning, run } = useComparisonRun(
    courseId,
    handleComplete,
  );

  // A run stored earlier this session means Evaluate has something to show
  // even before a new question is asked.
  const storedRun = getRun(courseId);
  const hasAnswered = Object.values(states).some((state) => state.status !== 'idle');

  const handleAsk = useCallback(
    (question: string) => {
      // Suggestions now come from this course's own examples, which have no
      // predefined-comparison id, so every run is recorded against its question
      // text. Results aggregation already falls back to that wording.
      void run(question, null);
    },
    [run],
  );

  const settled =
    hasAnswered &&
    !isRunning &&
    Object.values(states).every((state) => state.status !== 'loading');

  return (
    <div className="ui-stack ui-stack--section">
      <PageHeader
        title="Compare AI Responses"
        description="Ask one course question and compare how different AI approaches respond."
      />

      <QuestionAsk
        examples={suggestions.questions}
        examplesLabel={
          suggestions.source === 'course'
            ? 'Questions from this course'
            : 'Try an example'
        }
        examplesLoading={suggestions.loading}
        isRunning={isRunning}
        onAsk={handleAsk}
      />

      {!hasAnswered &&
        (storedRun ? (
          /* A comparison from earlier this session is still waiting to be
             rated. Saying "no question asked yet" next to a link to it was a
             contradiction. */
          <EmptyState
            illustration="contribute"
            title="Ask another question"
            description={`You last asked “${storedRun.question}”. Ask something new above, or go back and rate those answers.`}
            action={
              <LinkButton
                to={studentCoursePath(courseId, 'evaluate')}
                variant="secondary"
                iconRight="forward"
              >
                Rate your last comparison
              </LinkButton>
            }
          />
        ) : (
          <EmptyState
            illustration="contribute"
            title="No question asked yet"
            description="Type a question above, or pick one of the suggestions, to see four different answers side by side."
          />
        ))}

      {hasAnswered && (
        <>
          <p className="compare__question" role="status" aria-label="Active question">
            <span className="compare__question-label">You asked</span>
            {activeQuestion}
          </p>

          <section className="compare__grid" aria-label="Responses">
            {APPROACHES.map((approach, index) => (
              <ApproachCard
                key={approach.key}
                label={approach.label}
                description={approach.description}
                marker={String.fromCharCode(65 + index)}
                state={states[approach.key]}
              />
            ))}
          </section>

          {/* Evaluation is offered only once every approach has finished, so a
              student never rates a half-loaded set. */}
          <div className="compare__actions">
            {settled ? (
              <LinkButton
                to={studentCoursePath(courseId, 'evaluate')}
                variant="primary"
                iconRight="forward"
              >
                Evaluate these responses
              </LinkButton>
            ) : (
              <p className="compare__actions-hint" role="status">
                Waiting for all four responses…
              </p>
            )}
          </div>
        </>
      )}
    </div>
  );
}
