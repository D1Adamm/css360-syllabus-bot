import { useMemo } from 'react';
import comparisonData from '../../data/comparisonData.json';
import { ModelBarChart } from '../../components/ModelBarChart';
import { Button, LinkButton } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { EmptyState } from '../../components/ui/EmptyState';
import { PageHeader } from '../../components/ui/PageHeader';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { useCourseId } from '../../context/CourseContext';
import { useEvaluations } from '../../hooks/useEvaluations';
import { toUserMessage } from '../../lib/errorMessages';
import {
  professorCourseHomePath,
  professorCoursePath,
} from '../../lib/roleRoutes';
import type { ComparisonRecord } from '../../types';
import { exportEvaluationsJson } from '../../utils/exportData';
import {
  countHallucinationFlags,
  extractRecentComments,
  formatTopModels,
  getModelLabel,
  getTotalHallucinationFlags,
  getUniqueQuestionCount,
  groupByQuestion,
  MODEL_KEYS,
  tallyCriterion,
} from '../../utils/evaluationUtils';

/*
 * Kept solely to resolve question wording for evaluations recorded before
 * ratings carried their own `questionText`. It is a backward-compatibility
 * lookup, not a source of course content — nothing student-facing reads it.
 */
const comparisons = comparisonData as ComparisonRecord[];

/**
 * Aggregate student evaluation results.
 *
 * Cut down to the four questions a professor actually has: which approach did
 * students prefer, how did the approaches compare, where did students report
 * problems, and how much data is this based on. The previous page answered
 * those alongside eight summary tiles, six charts, and the same per-question
 * data rendered twice — once as a table and once as cards.
 */
export function ProfessorResultsPage() {
  const courseId = useCourseId();
  const { evaluations, loading, error } = useEvaluations();

  /*
   * Each criterion is tallied with its own denominator.
   *
   * "Closest to the syllabus" was retired from the student form, so a course
   * whose ratings are all recent has nobody who answered it. Charting that
   * against the evaluation total would draw four bars at 0% and read as
   * students having rejected every approach, which is the opposite of what
   * happened — they were never asked. It is charted only where there are
   * answers, and labelled with how many.
   */
  const preferred = useMemo(
    () => tallyCriterion(evaluations, 'preferredModel'),
    [evaluations],
  );
  const accurate = useMemo(
    () => tallyCriterion(evaluations, 'mostAccurate'),
    [evaluations],
  );
  const grounded = useMemo(
    () => tallyCriterion(evaluations, 'bestGrounded'),
    [evaluations],
  );
  const flags = useMemo(() => countHallucinationFlags(evaluations), [evaluations]);
  const perQuestion = useMemo(
    () => groupByQuestion(evaluations, comparisons),
    [evaluations],
  );
  const comments = useMemo(
    () => extractRecentComments(evaluations, comparisons),
    [evaluations],
  );

  const total = evaluations.length;
  const totalFlags = getTotalHallucinationFlags(evaluations);

  if (loading) {
    return (
      <div className="ui-stack ui-stack--loose">
        <PageHeader title="Results" description="What your students reported." />
        <p className="ui-text-muted" role="status" aria-live="polite">
          Loading results…
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="ui-stack ui-stack--loose">
        <PageHeader title="Results" description="What your students reported." />
        <Callout tone="danger" title="Results unavailable">
          {toUserMessage(new Error(error), {
            audience: 'professor',
            context: 'evaluation-load',
          }).message}
        </Callout>
      </div>
    );
  }

  if (total === 0) {
    return (
      <div className="ui-stack ui-stack--loose">
        <PageHeader title="Results" description="What your students reported." />
        <EmptyState
          illustration="empty-course"
          size="full"
          title="No evaluations yet"
          description="Results appear once students compare answers for this course and rate them. Reviewing examples doesn't produce results on its own — students have to use the course."
          action={
            <>
              <LinkButton
                to={professorCoursePath(courseId, 'invite')}
                variant="primary"
                iconLeft="students"
              >
                Invite students
              </LinkButton>
              <LinkButton to={professorCourseHomePath(courseId)} variant="tertiary">
                Back to overview
              </LinkButton>
            </>
          }
        />
      </div>
    );
  }

  return (
    <div className="ui-stack ui-stack--section">
      <PageHeader
        title="Results"
        description="What your students reported when they compared and rated responses."
        actions={
          <Button
            variant="tertiary"
            iconLeft="copy"
            onClick={() => exportEvaluationsJson(evaluations)}
          >
            Download results
          </Button>
        }
      />

      <section className="results-headline">
        <div className="results-headline__main">
          <p className="results-headline__label">Students preferred</p>
          <p className="results-headline__value">{formatTopModels(preferred.counts)}</p>
        </div>
        <dl className="results-headline__meta">
          <div>
            <dt>Evaluations</dt>
            <dd>{total}</dd>
          </div>
          <div>
            <dt>Questions covered</dt>
            <dd>{getUniqueQuestionCount(evaluations)}</dd>
          </div>
          <div>
            <dt>Problems reported</dt>
            <dd>{totalFlags}</dd>
          </div>
        </dl>
      </section>

      <section className="ui-stack ui-stack--snug">
        <SectionHeader
          title="How the approaches compared"
          description="Share of evaluations where students chose each approach."
          divider
        />
        <div className="results-charts">
          <ModelBarChart
            id="chart-preference"
            title="Preferred overall"
            counts={preferred.counts}
            total={preferred.answered}
          />
          <ModelBarChart
            id="chart-accuracy"
            title="Most accurate"
            counts={accurate.counts}
            total={accurate.answered}
          />
          {grounded.answered > 0 && (
            <ModelBarChart
              id="chart-grounding"
              title="Closest to the syllabus"
              counts={grounded.counts}
              total={grounded.answered}
            />
          )}
        </div>
        {grounded.answered > 0 && grounded.answered < total && (
          <p className="ui-text-xs ui-text-muted">
            Closest to the syllabus was asked of earlier evaluations only:{' '}
            {grounded.answered} of {total} answered it.
          </p>
        )}
      </section>

      {totalFlags > 0 && (
        <section className="ui-stack ui-stack--snug">
          <SectionHeader
            title="Where students reported problems"
            description="Answers flagged as containing something the syllabus doesn't support."
            divider
          />
          <ModelBarChart
            id="chart-flags"
            title="Flagged answers"
            counts={flags}
            total={totalFlags}
          />
        </section>
      )}

      <section className="ui-stack ui-stack--snug">
        <SectionHeader title="By question" divider />
        <ul className="question-results" aria-label="Results by question">
          {perQuestion.map((result) => {
            const flagged = MODEL_KEYS.filter(
              (key) => result.hallucinationFlags[key] > 0,
            );
            return (
              <li key={result.comparisonId} className="question-result">
                <p className="question-result__question">{result.question}</p>
                <p className="question-result__meta">
                  {result.evaluationCount} evaluation
                  {result.evaluationCount === 1 ? '' : 's'} · preferred{' '}
                  <strong>{result.mostPreferred}</strong>
                  {flagged.length > 0 && (
                    <>
                      {' '}
                      · flagged:{' '}
                      {flagged
                        .map(
                          (key) =>
                            `${getModelLabel(key)} (${result.hallucinationFlags[key]})`,
                        )
                        .join(', ')}
                    </>
                  )}
                </p>
              </li>
            );
          })}
        </ul>
      </section>

      {comments.length > 0 && (
        <section className="ui-stack ui-stack--snug">
          <SectionHeader title="What students said" divider />
          <ul className="result-comments">
            {comments.map((item) => (
              <li key={item.id} className="result-comment">
                <blockquote>{item.comment}</blockquote>
                <p className="result-comment__meta">
                  {item.question} · preferred {item.preferredModel}
                </p>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
