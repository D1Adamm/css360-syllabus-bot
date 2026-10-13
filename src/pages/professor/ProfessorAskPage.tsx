import { APPROACHES } from '../../components/compare/approaches';
import { ApproachCard } from '../../components/compare/ApproachCard';
import { QuestionAsk } from '../../components/compare/QuestionAsk';
import { Callout } from '../../components/ui/Callout';
import { EmptyState } from '../../components/ui/EmptyState';
import { PageHeader } from '../../components/ui/PageHeader';
import { useCourseId } from '../../context/CourseContext';
import { useSession } from '../../context/SessionContext';
import { isAdmin } from '../../context/session';
import { useQuestionSuggestions } from '../../hooks/useQuestionSuggestions';
import { useStaffComparison } from '../../hooks/useStaffComparison';

/**
 * Ask questions: a course's staff trying the four approaches themselves.
 *
 * The student Compare page's question box and answer cards, inside the
 * professor course area, for the course's assigned instructors and for
 * administrators. No class code and no participant session: the generation
 * routes admit course staff directly, and they write nothing. There is no
 * Evaluate step, and the comparison is never stored, so a staff question
 * cannot reach the course's examples, contributions or evaluation results.
 */
export function ProfessorAskPage() {
  const courseId = useCourseId();
  const { session } = useSession();
  const suggestions = useQuestionSuggestions(courseId);
  const { states, question, isRunning, ask } = useStaffComparison(
    courseId,
    isAdmin(session) ? 'admin' : 'professor',
  );
  const asked = question !== '';

  return (
    <div className="ui-stack ui-stack--section">
      <PageHeader
        title="Ask questions"
        description="Ask your own question about this course and compare how the four approaches answer it."
      />

      <Callout tone="info" title="Staff testing — not saved">
        These questions and answers are not saved. They are not added to this
        course&apos;s examples, student contributions or evaluation results.
      </Callout>

      <QuestionAsk
        examples={suggestions.questions}
        examplesLabel={
          suggestions.source === 'course' ? 'Questions from this course' : 'Try an example'
        }
        examplesLoading={suggestions.loading}
        isRunning={isRunning}
        onAsk={ask}
      />

      {!asked && (
        <EmptyState
          illustration="contribute"
          title="No question asked yet"
          description="Type a question above, or pick a suggestion, to see Base, RAG, Fine-Tuned and Fine-Tuned + RAG side by side."
        />
      )}

      {asked && (
        <>
          <p className="compare__question" role="status" aria-label="Active question">
            <span className="compare__question-label">You asked</span>
            {question}
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
        </>
      )}
    </div>
  );
}
