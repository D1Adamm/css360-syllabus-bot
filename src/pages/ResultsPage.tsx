import { useMemo } from 'react';
import { Link } from 'react-router-dom';
import comparisonData from '../data/comparisonData.json';
import { ModelBarChart } from '../components/ModelBarChart';
import { PageHeader } from '../components/PageHeader';
import { useCourseId } from '../context/CourseContext';
import { useEvaluations } from '../hooks/useEvaluations';
import { coursePagePath } from '../lib/courseRoutes';
import type { ComparisonRecord } from '../types';
import { exportEvaluationsJson } from '../utils/exportData';
import {
  countByField,
  countHallucinationFlags,
  extractRecentComments,
  formatEvaluationDate,
  formatTopModels,
  getModelLabel,
  getTotalEvaluationCount,
  getTotalHallucinationFlags,
  getUniqueQuestionCount,
  groupByQuestion,
  MODEL_KEYS,
} from '../utils/evaluationUtils';

const comparisons = comparisonData as ComparisonRecord[];

export function ResultsPage() {
  const courseId = useCourseId();
  const {
    evaluations,
    loading,
    error,
    saving,
    saveError,
    deleteAllEvaluations,
    clearSaveError,
  } = useEvaluations();

  const totalCount = getTotalEvaluationCount(evaluations);
  const uniqueQuestions = getUniqueQuestionCount(evaluations);

  const preferredCounts = useMemo(
    () => countByField(evaluations, 'preferredModel'),
    [evaluations],
  );
  const accurateCounts = useMemo(
    () => countByField(evaluations, 'mostAccurate'),
    [evaluations],
  );
  const helpfulCounts = useMemo(
    () => countByField(evaluations, 'mostHelpful'),
    [evaluations],
  );
  const conciseCounts = useMemo(
    () => countByField(evaluations, 'mostConcise'),
    [evaluations],
  );
  const groundedCounts = useMemo(
    () => countByField(evaluations, 'bestGrounded'),
    [evaluations],
  );
  const hallucinationCounts = useMemo(
    () => countHallucinationFlags(evaluations),
    [evaluations],
  );

  const perQuestionResults = useMemo(
    () => groupByQuestion(evaluations, comparisons),
    [evaluations],
  );

  const recentComments = useMemo(
    () => extractRecentComments(evaluations, comparisons),
    [evaluations],
  );

  const totalHallucinationFlags = getTotalHallucinationFlags(evaluations);

  async function handleReset() {
    const confirmed = window.confirm(
      'Delete all shared evaluation data? This cannot be undone. User-created seed examples will not be affected.',
    );
    if (confirmed) {
      clearSaveError();
      await deleteAllEvaluations();
    }
  }

  if (loading) {
    return (
      <>
        <PageHeader
          title="Results"
          description="View aggregated evaluation results calculated from ratings in the shared dataset."
        />
        <p className="results-status" role="status" aria-live="polite">
          Loading shared evaluations…
        </p>
      </>
    );
  }

  if (error) {
    return (
      <>
        <PageHeader
          title="Results"
          description="View aggregated evaluation results calculated from ratings in the shared dataset."
        />
        <p className="results-status results-status--error" role="alert">
          Could not load shared evaluations: {error}
        </p>
      </>
    );
  }

  if (totalCount === 0) {
    return (
      <>
        <PageHeader
          title="Results"
          description="View aggregated evaluation results calculated from ratings in the shared dataset."
        />

        <aside className="results-notice" aria-label="Results notice">
          <p>
            <strong>Shared prototype results:</strong> this page summarizes evaluations stored
            in Firebase Realtime Database and shared across browsers and devices.
          </p>
        </aside>

        <section className="results-empty" aria-live="polite">
          <h2 className="results-empty__title">No evaluations yet</h2>
          <p className="results-empty__text">
            No evaluations for this course yet. Complete at least one evaluation on the Model
            Comparison workflow to see aggregated results here. Each evaluation is saved under
            this course only.
          </p>
          <div className="results-empty__actions">
            <Link to={coursePagePath(courseId, 'compare')} className="button-link button-link--primary">
              Go to Model Comparison
            </Link>
            <Link to={coursePagePath(courseId, 'evaluate')} className="button-link button-link--secondary">
              Start evaluating
            </Link>
          </div>
        </section>
      </>
    );
  }

  return (
    <>
      <PageHeader
        title="Results"
        description="View aggregated evaluation results calculated from ratings in the shared dataset."
      />

      <aside className="results-notice" aria-label="Results notice">
        <p>
          <strong>Shared prototype results:</strong> this page summarizes evaluations stored in
          Firebase Realtime Database and shared across browsers and devices.
        </p>
      </aside>

      {saveError && (
        <p className="results-status results-status--error" role="alert">
          {saveError}
        </p>
      )}

      <section className="results-summary" aria-labelledby="results-summary-title">
        <h2 id="results-summary-title" className="results-summary__title">
          Summary metrics
        </h2>
        <dl className="results-summary__grid">
          <div className="results-summary__card">
            <dt className="results-summary__label">Total evaluations</dt>
            <dd className="results-summary__value">{totalCount}</dd>
          </div>
          <div className="results-summary__card">
            <dt className="results-summary__label">Unique questions evaluated</dt>
            <dd className="results-summary__value">{uniqueQuestions}</dd>
          </div>
          <div className="results-summary__card">
            <dt className="results-summary__label">Most preferred model</dt>
            <dd className="results-summary__value results-summary__value--text">
              {formatTopModels(preferredCounts)}
            </dd>
          </div>
          <div className="results-summary__card">
            <dt className="results-summary__label">Most accurate model</dt>
            <dd className="results-summary__value results-summary__value--text">
              {formatTopModels(accurateCounts)}
            </dd>
          </div>
          <div className="results-summary__card">
            <dt className="results-summary__label">Most helpful model</dt>
            <dd className="results-summary__value results-summary__value--text">
              {formatTopModels(helpfulCounts)}
            </dd>
          </div>
          <div className="results-summary__card">
            <dt className="results-summary__label">Most concise model</dt>
            <dd className="results-summary__value results-summary__value--text">
              {formatTopModels(conciseCounts)}
            </dd>
          </div>
          <div className="results-summary__card">
            <dt className="results-summary__label">Best grounded model</dt>
            <dd className="results-summary__value results-summary__value--text">
              {formatTopModels(groundedCounts)}
            </dd>
          </div>
          <div className="results-summary__card">
            <dt className="results-summary__label">Total hallucination flags</dt>
            <dd className="results-summary__value">{totalHallucinationFlags}</dd>
          </div>
        </dl>
      </section>

      <section className="results-charts" aria-labelledby="results-charts-title">
        <h2 id="results-charts-title" className="results-charts__title">
          Visual summaries
        </h2>
        <div className="results-charts__grid">
          <ModelBarChart
            id="chart-preference"
            title="Overall preference"
            counts={preferredCounts}
            total={totalCount}
          />
          <ModelBarChart
            id="chart-accuracy"
            title="Accuracy"
            counts={accurateCounts}
            total={totalCount}
          />
          <ModelBarChart
            id="chart-grounding"
            title="Grounding"
            counts={groundedCounts}
            total={totalCount}
          />
          <ModelBarChart
            id="chart-hallucination"
            title="Hallucination flags"
            counts={hallucinationCounts}
            total={totalHallucinationFlags}
          />
          <ModelBarChart
            id="chart-helpfulness"
            title="Helpfulness"
            counts={helpfulCounts}
            total={totalCount}
          />
          <ModelBarChart
            id="chart-conciseness"
            title="Conciseness"
            counts={conciseCounts}
            total={totalCount}
          />
        </div>
      </section>

      <section className="results-per-question" aria-labelledby="results-per-question-title">
        <h2 id="results-per-question-title" className="results-per-question__title">
          Results by question
        </h2>

        <div className="results-per-question__table-wrapper">
          <table className="results-per-question__table">
            <caption className="visually-hidden">
              Evaluation results grouped by comparison question
            </caption>
            <thead>
              <tr>
                <th scope="col">Question</th>
                <th scope="col">Evaluations</th>
                <th scope="col">Most preferred</th>
                <th scope="col">Most accurate</th>
                <th scope="col">Most grounded</th>
                <th scope="col">Hallucination flags</th>
              </tr>
            </thead>
            <tbody>
              {perQuestionResults.map((result) => (
                <tr key={result.comparisonId}>
                  <td>
                    <span className="results-per-question__question">{result.question}</span>
                    <span className="results-per-question__category">{result.category}</span>
                  </td>
                  <td>{result.evaluationCount}</td>
                  <td>{result.mostPreferred}</td>
                  <td>{result.mostAccurate}</td>
                  <td>{result.mostGrounded}</td>
                  <td>
                    <ul className="results-per-question__flags">
                      {MODEL_KEYS.filter((key) => result.hallucinationFlags[key] > 0).map(
                        (key) => (
                          <li key={key}>
                            {getModelLabel(key)}: {result.hallucinationFlags[key]}
                          </li>
                        ),
                      )}
                      {MODEL_KEYS.every((key) => result.hallucinationFlags[key] === 0) && (
                        <li>None</li>
                      )}
                    </ul>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <ul className="results-per-question__cards" aria-label="Results by question cards">
          {perQuestionResults.map((result) => (
            <li key={result.comparisonId} className="results-per-question__card">
              <h3 className="results-per-question__card-question">{result.question}</h3>
              <p className="results-per-question__card-category">{result.category}</p>
              <dl className="results-per-question__card-meta">
                <div>
                  <dt>Evaluations</dt>
                  <dd>{result.evaluationCount}</dd>
                </div>
                <div>
                  <dt>Most preferred</dt>
                  <dd>{result.mostPreferred}</dd>
                </div>
                <div>
                  <dt>Most accurate</dt>
                  <dd>{result.mostAccurate}</dd>
                </div>
                <div>
                  <dt>Most grounded</dt>
                  <dd>{result.mostGrounded}</dd>
                </div>
                <div>
                  <dt>Hallucination flags</dt>
                  <dd>
                    {MODEL_KEYS.filter((key) => result.hallucinationFlags[key] > 0)
                      .map((key) => `${getModelLabel(key)} (${result.hallucinationFlags[key]})`)
                      .join(', ') || 'None'}
                  </dd>
                </div>
              </dl>
            </li>
          ))}
        </ul>
      </section>

      {recentComments.length > 0 && (
        <section className="results-comments" aria-labelledby="results-comments-title">
          <h2 id="results-comments-title" className="results-comments__title">
            Recent evaluation notes
          </h2>
          <ul className="results-comments__list">
            {recentComments.map((item) => (
              <li key={item.id} className="results-comments__item">
                <p className="results-comments__question">{item.question}</p>
                <p className="results-comments__preferred">
                  Preferred model: <strong>{item.preferredModel}</strong>
                </p>
                <blockquote className="results-comments__quote">{item.comment}</blockquote>
                <time className="results-comments__date" dateTime={item.createdAt}>
                  {formatEvaluationDate(item.createdAt)}
                </time>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="results-actions" aria-label="Results actions">
        <button
          type="button"
          className="results-actions__export"
          onClick={() => exportEvaluationsJson(evaluations)}
        >
          Export evaluations as JSON
        </button>
        <button
          type="button"
          className="results-actions__reset"
          onClick={handleReset}
          disabled={saving}
        >
          {saving ? 'Deleting evaluations…' : 'Delete all evaluation data'}
        </button>
      </section>
    </>
  );
}
