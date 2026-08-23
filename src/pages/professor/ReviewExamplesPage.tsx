import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import { ReviewListRow } from '../../components/review/ReviewListRow';
import { ReviewQueueCard } from '../../components/review/ReviewQueueCard';
import { Button } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { ConfirmDialog } from '../../components/ui/ConfirmDialog';
import { ErrorState } from '../../components/ui/ErrorState';
import { EmptyState } from '../../components/ui/EmptyState';
import { Icon } from '../../components/ui/Icon';
import { PageHeader } from '../../components/ui/PageHeader';
import { useCourseId } from '../../context/CourseContext';
import { useCourseMetadata } from '../../hooks/useCourseMetadata';
import { useExampleReview, type ReviewDraft } from '../../hooks/useExampleReview';
import type { CourseSeedReviewRecord } from '../../lib/api';
import { toUserMessage } from '../../lib/errorMessages';
import {
  exampleAnswer,
  exampleQuestion,
  resolveExampleStatus,
} from '../../lib/exampleCounts';
import { findDuplicateExampleIds } from '../../lib/exampleDuplicates';
import {
  describeStarterGeneration,
  getStarterGeneration,
} from '../../lib/starterSeedGeneration';

type ReviewFilter = 'pending' | 'approved' | 'rejected' | 'edited' | 'all';
type ViewMode = 'list' | 'card';

const FILTERS: { id: ReviewFilter; label: string }[] = [
  { id: 'pending', label: 'Awaiting review' },
  { id: 'approved', label: 'Approved' },
  { id: 'edited', label: 'Edited' },
  { id: 'rejected', label: 'Rejected' },
  { id: 'all', label: 'All' },
];

const ALL_CATEGORIES = 'All categories';

function matchesFilter(status: string, filter: ReviewFilter): boolean {
  if (filter === 'all') {
    return true;
  }
  if (filter === 'pending') {
    return status !== 'approved' && status !== 'rejected' && status !== 'edited';
  }
  return status === filter;
}

function isReviewed(status: string): boolean {
  return status === 'approved' || status === 'rejected' || status === 'edited';
}

function seedIdOf(example: CourseSeedReviewRecord): string {
  return String(example.id || '').trim();
}

function matchesSearch(example: CourseSeedReviewRecord, needle: string): boolean {
  if (!needle) {
    return true;
  }
  const haystack = [
    exampleQuestion(example),
    exampleAnswer(example),
    String(example.category || ''),
    String(example.sourceSection || ''),
  ]
    .join(' ')
    .toLowerCase();
  return haystack.includes(needle);
}

/**
 * The review queue.
 *
 * Two modes for the same decision. Card view reads one example at a time and
 * is the right shape for a careful pass over a handful. List view exists
 * because a generated batch is fifty, and clicking Next fifty times is not
 * review, it is data entry — so the list puts every example, and every action
 * on it, on one scrollable page. Neither mode approves anything on its own:
 * generation and review stay separate, and every status change here is a
 * professor's explicit click.
 *
 * List view is the default for that reason. Card view is one click away and
 * unchanged.
 */
export function ReviewExamplesPage() {
  const courseId = useCourseId();
  const {
    examples,
    counts,
    loading,
    error,
    busyId,
    bulkBusy,
    actionMessage,
    actionFailed,
    reload,
    approve,
    reject,
    saveEdit,
    reviewMany,
  } = useExampleReview(courseId);

  /*
   * Whether this course's starter examples are still being written.
   *
   * Read from the course's own metadata, which the generation job updates as
   * it goes. That is why a refresh keeps showing "generating": the answer is
   * durable and course-scoped, and this page only reads it.
   */
  const { metadata } = useCourseMetadata(courseId);
  const generation = getStarterGeneration(metadata);
  const generationMessage = describeStarterGeneration(generation.state);

  const fieldId = useId();
  const [view, setView] = useState<ViewMode>('list');
  const [filter, setFilter] = useState<ReviewFilter>('pending');
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState(ALL_CATEGORIES);
  const [hideReviewed, setHideReviewed] = useState(false);
  const [index, setIndex] = useState(0);

  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  /* The inline-edit draft, held here so it survives the row unmounting when an
     optimistic status change takes it out of the current filter. */
  const [edit, setEdit] = useState<
    { seedId: string; question: string; answer: string } | null
  >(null);
  const [focusedId, setFocusedId] = useState<string | null>(null);
  const [confirmingBulkReject, setConfirmingBulkReject] = useState(false);

  const needle = search.trim().toLowerCase();

  const categories = useMemo(() => {
    const names = new Set<string>();
    for (const example of examples) {
      const name = String(example.category || '').trim();
      if (name) {
        names.add(name);
      }
    }
    return [...names].sort((a, b) => a.localeCompare(b));
  }, [examples]);

  const visible = useMemo(
    () =>
      examples.filter((example) => {
        const status = resolveExampleStatus(example);
        if (!matchesFilter(status, filter)) {
          return false;
        }
        if (hideReviewed && isReviewed(status)) {
          return false;
        }
        if (
          category !== ALL_CATEGORIES &&
          String(example.category || '').trim() !== category
        ) {
          return false;
        }
        return matchesSearch(example, needle);
      }),
    [examples, filter, hideReviewed, category, needle],
  );

  const duplicateIds = useMemo(() => findDuplicateExampleIds(examples), [examples]);

  const reviewed = counts.approved + counts.rejected + counts.edited;
  const busy = busyId !== null || bulkBusy;

  // Approving the last item in a filter would otherwise leave the queue past
  // its end.
  useEffect(() => {
    setIndex((current) => Math.min(current, Math.max(0, visible.length - 1)));
  }, [visible.length]);

  useEffect(() => {
    setIndex(0);
  }, [filter]);

  /*
   * Keyboard focus survives the row it was on leaving the list.
   *
   * Approving under "Awaiting review" removes that row, and dropping focus
   * back to the top would undo the work of scrolling to it. Focus moves to
   * whatever now occupies the same position instead.
   */
  const focusedIndexRef = useRef(0);
  const focusedIndex = focusedId
    ? visible.findIndex((example) => seedIdOf(example) === focusedId)
    : -1;

  useEffect(() => {
    if (focusedIndex >= 0) {
      focusedIndexRef.current = focusedIndex;
      return;
    }
    if (view !== 'list' || visible.length === 0 || focusedId === null) {
      return;
    }
    const next = visible[Math.min(focusedIndexRef.current, visible.length - 1)];
    setFocusedId(next ? seedIdOf(next) : null);
  }, [focusedIndex, visible, focusedId, view]);

  const current = visible[index];

  const goPrevious = useCallback(() => {
    setIndex((value) => Math.max(0, value - 1));
  }, []);

  const goNext = useCallback(() => {
    setIndex((value) => Math.min(visible.length - 1, value + 1));
  }, [visible.length]);

  const moveFocus = useCallback(
    (delta: number) => {
      if (visible.length === 0) {
        return;
      }
      const from = focusedIndex >= 0 ? focusedIndex : -1;
      const next = Math.min(
        visible.length - 1,
        Math.max(0, from < 0 ? 0 : from + delta),
      );
      const example = visible[next];
      if (example) {
        setFocusedId(seedIdOf(example));
      }
    },
    [focusedIndex, visible],
  );

  const startEdit = useCallback(
    (seedId: string) => {
      const example = examples.find((item) => seedIdOf(item) === seedId);
      if (!example) {
        return;
      }
      setExpandedIds((ids) => new Set(ids).add(seedId));
      setEdit({
        seedId,
        question: exampleQuestion(example),
        answer: exampleAnswer(example),
      });
    },
    [examples],
  );

  /*
   * Keyboard shortcuts.
   *
   * Ignored while typing, without exception: `a` inside the inline editor has
   * to be the letter a, not an approval. Every shortcut here duplicates a
   * button that is visible on the row, so nothing is reachable only this way.
   */
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable)
      ) {
        return;
      }
      if (event.metaKey || event.ctrlKey || event.altKey) {
        return;
      }

      if (view === 'card') {
        if (busy || !current) {
          return;
        }
        if (event.key === 'ArrowLeft' || event.key === 'k') {
          goPrevious();
        } else if (event.key === 'ArrowRight' || event.key === 'j') {
          goNext();
        } else if (event.key === 'a') {
          void approve(seedIdOf(current));
        } else if (event.key === 'r') {
          void reject(seedIdOf(current));
        }
        return;
      }

      if (event.key === 'ArrowDown' || event.key === 'j') {
        event.preventDefault();
        moveFocus(1);
        return;
      }
      if (event.key === 'ArrowUp' || event.key === 'k') {
        event.preventDefault();
        moveFocus(-1);
        return;
      }

      const focused = visible.find((example) => seedIdOf(example) === focusedId);
      if (!focused || busy) {
        return;
      }
      if (event.key === 'a') {
        void approve(seedIdOf(focused));
      } else if (event.key === 'r') {
        void reject(seedIdOf(focused));
      } else if (event.key === 'e') {
        startEdit(seedIdOf(focused));
      }
    }

    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [
    approve,
    busy,
    current,
    focusedId,
    goNext,
    goPrevious,
    moveFocus,
    reject,
    startEdit,
    view,
    visible,
  ]);

  const handleSaveEdit = useCallback(
    (seedId: string, draft: ReviewDraft) => {
      void saveEdit(seedId, draft);
    },
    [saveEdit],
  );

  /*
   * Inline save closes the editor only once the write has landed. Closing it
   * optimistically would show the new wording next to an unchanged database.
   */
  const handleRowSaveEdit = useCallback(
    (seedId: string, draft: ReviewDraft) => {
      void saveEdit(seedId, draft).then((saved) => {
        if (saved) {
          setEdit((current) => (current?.seedId === seedId ? null : current));
        }
      });
    },
    [saveEdit],
  );

  const toggleExpanded = useCallback((seedId: string) => {
    setExpandedIds((ids) => {
      const next = new Set(ids);
      if (next.has(seedId)) {
        next.delete(seedId);
      } else {
        next.add(seedId);
      }
      return next;
    });
  }, []);

  const toggleSelected = useCallback((seedId: string) => {
    setSelectedIds((ids) => {
      const next = new Set(ids);
      if (next.has(seedId)) {
        next.delete(seedId);
      } else {
        next.add(seedId);
      }
      return next;
    });
  }, []);

  const expandAll = useCallback(() => {
    setExpandedIds(new Set(visible.map(seedIdOf).filter(Boolean)));
  }, [visible]);

  const collapseAll = useCallback(() => {
    setExpandedIds(new Set());
  }, []);

  const selectAllVisible = useCallback(() => {
    setSelectedIds(new Set(visible.map(seedIdOf).filter(Boolean)));
  }, [visible]);

  const clearSelection = useCallback(() => {
    setSelectedIds(new Set());
  }, []);

  /*
   * Bulk review keeps whatever failed selected, so a retry is one more click
   * and the professor can see exactly which examples did not go through.
   */
  const runBulk = useCallback(
    async (reviewStatus: 'approved' | 'rejected') => {
      const ids = [...selectedIds];
      const result = await reviewMany(ids, reviewStatus);
      setSelectedIds(new Set(result.failed));
    },
    [reviewMany, selectedIds],
  );

  const selectedCount = selectedIds.size;

  const viewToggle = (
    <div className="review-view-toggle" role="group" aria-label="View mode">
      <button
        type="button"
        className={
          view === 'card' ? 'review-view-toggle__option is-active' : 'review-view-toggle__option'
        }
        aria-pressed={view === 'card'}
        onClick={() => setView('card')}
      >
        Card view
      </button>
      <button
        type="button"
        className={
          view === 'list' ? 'review-view-toggle__option is-active' : 'review-view-toggle__option'
        }
        aria-pressed={view === 'list'}
        onClick={() => setView('list')}
      >
        List view
      </button>
    </div>
  );

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        title="Review Examples"
        description="Approve, edit, or reject the example questions collected for this course. Nothing is approved automatically."
        actions={
          <Button
            variant="tertiary"
            iconLeft="status"
            onClick={() => void reload()}
            disabled={loading || busy}
          >
            Refresh
          </Button>
        }
      />

      {/* When the load failed we know nothing about this course's examples.
          Showing the filter tabs with zeroes next to an error banner tells the
          reader two contradictory things, and the zeroes are the more
          believable one. So the error replaces the data entirely. */}
      {error ? (
        <ErrorState
          title="Examples unavailable"
          message={
            toUserMessage(new Error(error), {
              audience: 'professor',
              context: 'examples-load',
            }).message
          }
          onRetry={() => void reload()}
        />
      ) : (
        <>
          {actionMessage && (
            <Callout tone={actionFailed ? 'danger' : 'success'}>{actionMessage}</Callout>
          )}

          {/* Examples have started arriving and more are still coming. A professor
              looking at six examples has no other way to know that. When there are
              none yet the empty state below says the same thing more prominently,
              so this would only repeat it.

              A failure is deliberately not announced here: once there are examples
              to review, telling a professor something went wrong on a page full of
              work they can do is alarming and useless. The empty state covers the
              case where it actually matters. */}
          {generation.state === 'generating' && counts.total > 0 && generationMessage && (
            <Callout tone="info" title={generationMessage.title}>
              {generationMessage.detail}
            </Callout>
          )}

          <div className="review-controls">
            <div className="filter-tabs" role="tablist" aria-label="Filter examples">
              {FILTERS.map((item) => {
                const count =
                  item.id === 'pending'
                    ? counts.pending
                    : item.id === 'approved'
                      ? counts.approved
                      : item.id === 'edited'
                        ? counts.edited
                        : item.id === 'rejected'
                          ? counts.rejected
                          : counts.total;

                return (
                  <button
                    key={item.id}
                    type="button"
                    role="tab"
                    aria-selected={filter === item.id}
                    className={
                      filter === item.id
                        ? 'filter-tab filter-tab--active'
                        : 'filter-tab'
                    }
                    onClick={() => setFilter(item.id)}
                  >
                    {item.label}
                    <span className="filter-tab__count">{count}</span>
                  </button>
                );
              })}
            </div>

            {viewToggle}
          </div>

          {/* Two real numbers, not a progress bar. Nothing here knows how many
              examples a course "should" end up with, so a percentage would be
              invented. */}
          {counts.total > 0 && (
            <p className="review-progress" aria-live="polite">
              {reviewed} reviewed · {counts.pending} remaining
            </p>
          )}

          {loading ? (
            <p className="ui-text-muted" role="status" aria-live="polite">
              Loading examples…
            </p>
          ) : (
            <>
              {counts.total > 0 && (
                <div className="review-filters">
                  <div className="ui-field review-filters__search">
                    <label className="ui-field__label" htmlFor={`${fieldId}-search`}>
                      Search examples
                    </label>
                    <div className="ui-field__control">
                      <input
                        id={`${fieldId}-search`}
                        type="search"
                        value={search}
                        placeholder="Question or answer text…"
                        onChange={(event) => setSearch(event.target.value)}
                      />
                    </div>
                  </div>

                  {categories.length > 1 && (
                    <div className="ui-field">
                      <label className="ui-field__label" htmlFor={`${fieldId}-category`}>
                        Category
                      </label>
                      <div className="ui-field__control">
                        <select
                          id={`${fieldId}-category`}
                          value={category}
                          onChange={(event) => setCategory(event.target.value)}
                        >
                          <option value={ALL_CATEGORIES}>{ALL_CATEGORIES}</option>
                          {categories.map((name) => (
                            <option key={name} value={name}>
                              {name}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  )}

                  <div className="review-filters__toggle">
                    <input
                      type="checkbox"
                      id={`${fieldId}-hide-reviewed`}
                      checked={hideReviewed}
                      onChange={(event) => setHideReviewed(event.target.checked)}
                    />
                    <label htmlFor={`${fieldId}-hide-reviewed`}>Hide reviewed</label>
                  </div>
                </div>
              )}

              {visible.length === 0 && counts.total === 0 && generationMessage ? (
                /* An empty course whose examples are still being made, or whose
                   generation failed, is not "nothing collected yet" — that wording
                   reads as "your upload did nothing" and is what left a professor
                   staring at zero with no explanation. */
                <EmptyState
                  illustration="empty-course"
                  title={generationMessage.title}
                  description={generationMessage.detail}
                />
              ) : visible.length === 0 ? (
                <EmptyState
                  illustration="empty-course"
                  title={
                    needle || category !== ALL_CATEGORIES
                      ? 'No matching examples'
                      : filter === 'pending'
                        ? 'Nothing waiting for you'
                        : 'Nothing to show'
                  }
                  description={
                    counts.total === 0
                      ? 'No example questions have been collected for this course yet.'
                      : needle || category !== ALL_CATEGORIES
                        ? 'Try a different search or category.'
                        : filter === 'pending'
                          ? 'Every example has been reviewed. Switch filters to revisit your decisions.'
                          : 'No examples match this filter.'
                  }
                />
              ) : view === 'card' ? (
                <>
                  <div className="review-queue-bar">
                    <p className="review-queue-bar__position" aria-live="polite">
                      {index + 1} of {visible.length}
                    </p>
                    <div className="review-queue-bar__controls">
                      <Button
                        variant="tertiary"
                        size="sm"
                        iconLeft="previous"
                        onClick={goPrevious}
                        disabled={index === 0 || busy}
                      >
                        Previous
                      </Button>
                      <Button
                        variant="tertiary"
                        size="sm"
                        iconRight="next"
                        onClick={goNext}
                        disabled={index >= visible.length - 1 || busy}
                      >
                        Next
                      </Button>
                    </div>
                  </div>

                  {current && (
                    <ReviewQueueCard
                      key={seedIdOf(current) || String(index)}
                      example={current}
                      busy={busyId === seedIdOf(current)}
                      onApprove={(seedId) => void approve(seedId)}
                      onReject={(seedId) => void reject(seedId)}
                      onSaveEdit={handleSaveEdit}
                    />
                  )}

                  <p className="review-shortcuts">
                    <Icon name="info" size={13} />
                    <span>
                      <kbd>←</kbd> <kbd>→</kbd> to move · <kbd>a</kbd> approve ·{' '}
                      <kbd>r</kbd> reject
                    </span>
                  </p>
                </>
              ) : (
                <>
                  <div className="review-list-bar">
                    <p className="review-list-bar__count" aria-live="polite">
                      Showing {visible.length} of {counts.total}
                    </p>
                    <div className="review-list-bar__controls">
                      <Button size="sm" variant="ghost" onClick={expandAll}>
                        Expand all
                      </Button>
                      <Button size="sm" variant="ghost" onClick={collapseAll}>
                        Collapse all
                      </Button>
                      <Button size="sm" variant="ghost" onClick={selectAllVisible}>
                        Select all visible
                      </Button>
                    </div>
                  </div>

                  {selectedCount > 0 && (
                    <div className="review-bulk-bar" role="region" aria-label="Bulk actions">
                      <p className="review-bulk-bar__count" aria-live="polite">
                        {selectedCount} selected
                      </p>
                      <div className="review-bulk-bar__actions">
                        <Button
                          size="sm"
                          variant="secondary"
                          onClick={() => void runBulk('approved')}
                          loading={bulkBusy}
                          loadingLabel="Saving…"
                        >
                          Approve selected
                        </Button>
                        <Button
                          size="sm"
                          variant="tertiary"
                          onClick={() => setConfirmingBulkReject(true)}
                          disabled={bulkBusy}
                        >
                          Reject selected
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={clearSelection}
                          disabled={bulkBusy}
                        >
                          Clear selection
                        </Button>
                      </div>
                    </div>
                  )}

                  <ul className="review-list" aria-label="Examples awaiting your decision">
                    {visible.map((example, position) => {
                      const seedId = seedIdOf(example);
                      return (
                        <ReviewListRow
                          key={seedId || String(position)}
                          example={example}
                          position={position + 1}
                          total={visible.length}
                          expanded={expandedIds.has(seedId)}
                          selected={selectedIds.has(seedId)}
                          draft={
                            edit?.seedId === seedId
                              ? { question: edit.question, answer: edit.answer }
                              : null
                          }
                          focused={focusedId === seedId}
                          busy={busyId === seedId}
                          disabled={bulkBusy || (busyId !== null && busyId !== seedId)}
                          duplicate={duplicateIds.has(seedId)}
                          onToggleExpanded={toggleExpanded}
                          onToggleSelected={toggleSelected}
                          onFocusRow={setFocusedId}
                          onStartEdit={startEdit}
                          onDraftChange={(patch) =>
                            setEdit((current) =>
                              current?.seedId === seedId
                                ? { ...current, ...patch }
                                : current,
                            )
                          }
                          onCancelEdit={() => setEdit(null)}
                          onSaveEdit={handleRowSaveEdit}
                          onApprove={(id) => void approve(id)}
                          onReject={(id) => void reject(id)}
                        />
                      );
                    })}
                  </ul>

                  <p className="review-shortcuts">
                    <Icon name="info" size={13} />
                    <span>
                      <kbd>↑</kbd> <kbd>↓</kbd> to move · <kbd>a</kbd> approve ·{' '}
                      <kbd>r</kbd> reject · <kbd>e</kbd> edit
                    </span>
                  </p>
                </>
              )}
            </>
          )}
        </>
      )}

      <ConfirmDialog
        open={confirmingBulkReject}
        tone="danger"
        title={`Reject ${selectedCount} ${selectedCount === 1 ? 'example' : 'examples'}?`}
        description="Rejected examples are excluded from training. You can approve them again later from the Rejected filter."
        confirmLabel="Reject selected"
        onConfirm={() => {
          setConfirmingBulkReject(false);
          void runBulk('rejected');
        }}
        onCancel={() => setConfirmingBulkReject(false)}
      />
    </div>
  );
}
