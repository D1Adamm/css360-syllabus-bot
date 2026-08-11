import { useMemo, useState } from 'react';
import { Button } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { ConfirmDialog } from '../../components/ui/ConfirmDialog';
import { EmptyState } from '../../components/ui/EmptyState';
import { PageHeader } from '../../components/ui/PageHeader';
import { StatusPill } from '../../components/ui/StatusPill';
import { useCourseId } from '../../context/CourseContext';
import { useSeedExamples } from '../../hooks/useSeedExamples';
import {
  exportCompleteJsonl,
  exportFilteredJson,
  exportFilteredJsonl,
} from '../../utils/exportData';
import {
  ALL_ANSWER_TYPES,
  ALL_CATEGORIES,
  ALL_DIFFICULTIES,
  type AnswerTypeFilter,
  REVIEW_STATUS_FILTERS,
  type ReviewStatusFilter,
  calculateStatistics,
  filterSeeds,
  getSeedOriginLabel,
  getUniqueCategories,
  resolveSeedReviewStatus,
  type SortOption,
} from '../../utils/seedDataUtils';
import type { SeedExample } from '../../types';

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/**
 * The full dataset view.
 *
 * This is the old "Seed Dataset" page, now admin-only. Everything a student or
 * professor should never see lives here on purpose: validation component
 * scores, provenance, raw ids, origin, and the JSON/JSONL exports.
 */
export function AdminExamplesPage() {
  const courseId = useCourseId();
  const { seeds, loading, error, deleteSeed } = useSeedExamples();

  const [search, setSearch] = useState('');
  const [category, setCategory] = useState(ALL_CATEGORIES);
  const [difficulty, setDifficulty] = useState(ALL_DIFFICULTIES);
  const [answerType, setAnswerType] = useState<AnswerTypeFilter>(ALL_ANSWER_TYPES);
  const [reviewStatus, setReviewStatus] = useState<ReviewStatusFilter>('approved');
  const [sortBy, setSortBy] = useState<SortOption>('id-asc');
  const [expanded, setExpanded] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<SeedExample | null>(null);

  const categories = useMemo(() => getUniqueCategories(seeds), [seeds]);
  const stats = useMemo(() => calculateStatistics(seeds), [seeds]);

  const filtered = useMemo(
    () =>
      filterSeeds(seeds, {
        searchQuery: search,
        category,
        difficulty,
        answerType,
        reviewStatus,
        sortBy,
      }),
    [seeds, search, category, difficulty, answerType, reviewStatus, sortBy],
  );

  function clearFilters() {
    setSearch('');
    setCategory(ALL_CATEGORIES);
    setDifficulty(ALL_DIFFICULTIES);
    setAnswerType(ALL_ANSWER_TYPES);
    setReviewStatus('approved');
    setSortBy('id-asc');
  }

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        title="Examples"
        eyebrow="Admin"
        description={`Full dataset for ${courseId}, including validation detail and export.`}
      />

      {error && (
        <Callout tone="danger" title="Could not read examples">
          {error}
        </Callout>
      )}

      {loading ? (
        <p className="ui-text-muted" role="status" aria-live="polite">
          Loading examples…
        </p>
      ) : seeds.length === 0 ? (
        <EmptyState
          title="No examples stored"
          description="Nothing has been generated or contributed for this course yet."
        />
      ) : (
        <>
          <dl className="admin-stats" aria-label="Review status summary">
            {[
              { label: 'Total', value: stats.totalExamples },
              { label: 'Approved', value: stats.approvedCount },
              { label: 'Rejected', value: stats.rejectedCount },
              { label: 'Awaiting review', value: stats.generatedCount },
              { label: 'Edited', value: stats.editedCount },
            ].map((item) => (
              <div key={item.label} className="admin-stat">
                <dt>{item.label}</dt>
                <dd>{item.value}</dd>
              </div>
            ))}
          </dl>

          <div className="filter-tabs" role="tablist" aria-label="Filter by review status">
            {REVIEW_STATUS_FILTERS.map((item) => (
              <button
                key={item.id}
                type="button"
                role="tab"
                aria-selected={reviewStatus === item.id}
                className={
                  reviewStatus === item.id ? 'filter-tab filter-tab--active' : 'filter-tab'
                }
                onClick={() => setReviewStatus(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>

          <div className="admin-filters">
            <div className="ui-field">
              <label className="ui-field__label" htmlFor="admin-search">
                Search
              </label>
              <div className="ui-field__control">
                <input
                  id="admin-search"
                  type="search"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Question, answer, category, or section…"
                />
              </div>
            </div>

            <div className="ui-field">
              <label className="ui-field__label" htmlFor="admin-category">
                Category
              </label>
              <div className="ui-field__control">
                <select
                  id="admin-category"
                  value={category}
                  onChange={(event) => setCategory(event.target.value)}
                >
                  <option value={ALL_CATEGORIES}>{ALL_CATEGORIES}</option>
                  {categories.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="ui-field">
              <label className="ui-field__label" htmlFor="admin-difficulty">
                Difficulty
              </label>
              <div className="ui-field__control">
                <select
                  id="admin-difficulty"
                  value={difficulty}
                  onChange={(event) => setDifficulty(event.target.value)}
                >
                  <option value={ALL_DIFFICULTIES}>{ALL_DIFFICULTIES}</option>
                  {['Easy', 'Medium', 'Hard'].map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="ui-field">
              <label className="ui-field__label" htmlFor="admin-answer-type">
                Answer type
              </label>
              <div className="ui-field__control">
                <select
                  id="admin-answer-type"
                  value={answerType}
                  onChange={(event) =>
                    setAnswerType(event.target.value as AnswerTypeFilter)
                  }
                >
                  {[ALL_ANSWER_TYPES, 'Directly answered', 'Not directly answered'].map(
                    (item) => (
                      <option key={item} value={item}>
                        {item}
                      </option>
                    ),
                  )}
                </select>
              </div>
            </div>

            <div className="ui-field">
              <label className="ui-field__label" htmlFor="admin-sort">
                Sort by
              </label>
              <div className="ui-field__control">
                <select
                  id="admin-sort"
                  value={sortBy}
                  onChange={(event) => setSortBy(event.target.value as SortOption)}
                >
                  <option value="id-asc">ID ascending</option>
                  <option value="category-asc">Category A–Z</option>
                  <option value="difficulty">Difficulty</option>
                  <option value="question-asc">Question A–Z</option>
                </select>
              </div>
            </div>
          </div>

          <div className="admin-row admin-row--stacked">
            <div className="admin-row__main">
              <p className="admin-row__label">Export</p>
              <p className="ui-text-xs ui-text-muted">
                Downloads in the browser. Does not touch the server-side export.
              </p>
            </div>
            <div className="admin-row__actions">
              <Button size="sm" variant="secondary" onClick={() => exportFilteredJson(filtered)}>
                Filtered JSON
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => exportFilteredJsonl(filtered)}
              >
                Filtered JSONL
              </Button>
              <Button size="sm" variant="secondary" onClick={() => exportCompleteJsonl(seeds)}>
                Complete JSONL
              </Button>
            </div>
          </div>

          <p className="ui-text-sm ui-text-muted" aria-live="polite">
            Showing {filtered.length} of {seeds.length}
          </p>

          {filtered.length === 0 ? (
            <EmptyState
              title="No matching examples"
              description="Try a different search or filter combination."
              action={
                <Button variant="secondary" onClick={clearFilters}>
                  Clear filters
                </Button>
              }
            />
          ) : (
            <ul className="admin-rows" aria-label="Examples">
              {filtered.map((seed) => {
                const isOpen = expanded === seed.id;
                const components = seed.validation?.components;
                const unsupported = seed.validation?.unsupportedClaims ?? [];

                return (
                  <li key={seed.id} className="admin-row admin-row--stacked">
                    <div className="admin-row__main">
                      <div className="admin-example__labels">
                        <StatusPill
                          tone={
                            resolveSeedReviewStatus(seed) === 'approved'
                              ? 'success'
                              : resolveSeedReviewStatus(seed) === 'rejected'
                                ? 'danger'
                                : 'neutral'
                          }
                        >
                          {String(resolveSeedReviewStatus(seed))}
                        </StatusPill>
                        <span className="ui-text-xs ui-text-muted">
                          {getSeedOriginLabel(seed.origin)} · {seed.category} ·{' '}
                          {seed.difficulty}
                        </span>
                        {typeof seed.validation?.score === 'number' && (
                          <span className="ui-text-xs ui-text-muted">
                            validation {percent(seed.validation.score)}
                          </span>
                        )}
                      </div>

                      <p className="admin-example__question">{seed.instruction}</p>
                      <p className="admin-example__answer">
                        {isOpen
                          ? seed.response
                          : `${seed.response.slice(0, 140)}${seed.response.length > 140 ? '…' : ''}`}
                      </p>

                      {isOpen && (
                        <dl className="admin-example__meta">
                          <div>
                            <dt>ID</dt>
                            <dd>
                              <code>{seed.id}</code>
                            </dd>
                          </div>
                          <div>
                            <dt>Source section</dt>
                            <dd>{seed.sourceSection}</dd>
                          </div>
                          {seed.validation?.reason && (
                            <div>
                              <dt>Validation</dt>
                              <dd>{seed.validation.reason}</dd>
                            </div>
                          )}
                          {components && (
                            <div>
                              <dt>Components</dt>
                              <dd>
                                grounded {percent(components.grounded)}, correct{' '}
                                {percent(components.correct)}, clear{' '}
                                {percent(components.clear)}, useful{' '}
                                {percent(components.useful)}, wording{' '}
                                {percent(components.naturalStudentWording)}, category{' '}
                                {percent(components.categoryCorrect)}, not trivial{' '}
                                {percent(components.notTrivialOrTemporary)}
                              </dd>
                            </div>
                          )}
                          {unsupported.length > 0 && (
                            <div>
                              <dt>Unsupported claims</dt>
                              <dd>{unsupported.join('; ')}</dd>
                            </div>
                          )}
                          {seed.originalQuestion && (
                            <div>
                              <dt>Original question</dt>
                              <dd>{seed.originalQuestion}</dd>
                            </div>
                          )}
                        </dl>
                      )}
                    </div>

                    <div className="admin-row__actions">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => setExpanded(isOpen ? null : seed.id)}
                        aria-expanded={isOpen}
                      >
                        {isOpen ? 'Collapse' : 'Expand'}
                      </Button>
                      {seed.origin === 'user' && (
                        <Button
                          size="sm"
                          variant="ghost"
                          iconLeft="delete"
                          onClick={() => setPendingDelete(seed)}
                        >
                          Delete
                        </Button>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </>
      )}

      <ConfirmDialog
        open={pendingDelete !== null}
        tone="danger"
        title="Delete this example?"
        description={
          pendingDelete
            ? `"${pendingDelete.instruction}" will be removed from this course. This cannot be undone.`
            : ''
        }
        confirmLabel="Delete"
        onConfirm={() => {
          if (pendingDelete) {
            void deleteSeed(pendingDelete.id);
          }
          setPendingDelete(null);
        }}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  );
}
