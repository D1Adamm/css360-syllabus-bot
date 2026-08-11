import { useCallback, useEffect, useMemo, useState } from 'react';
import { ReviewQueueCard } from '../../components/review/ReviewQueueCard';
import { Button } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { ErrorState } from '../../components/ui/ErrorState';
import { EmptyState } from '../../components/ui/EmptyState';
import { Icon } from '../../components/ui/Icon';
import { PageHeader } from '../../components/ui/PageHeader';
import { useCourseId } from '../../context/CourseContext';
import { useExampleReview, type ReviewDraft } from '../../hooks/useExampleReview';
import { toUserMessage } from '../../lib/errorMessages';
import { resolveExampleStatus } from '../../lib/exampleCounts';

type ReviewFilter = 'pending' | 'approved' | 'rejected' | 'edited' | 'all';

const FILTERS: { id: ReviewFilter; label: string }[] = [
  { id: 'pending', label: 'Awaiting review' },
  { id: 'approved', label: 'Approved' },
  { id: 'edited', label: 'Edited' },
  { id: 'rejected', label: 'Rejected' },
  { id: 'all', label: 'All' },
];

function matchesFilter(status: string, filter: ReviewFilter): boolean {
  if (filter === 'all') {
    return true;
  }
  if (filter === 'pending') {
    return status !== 'approved' && status !== 'rejected' && status !== 'edited';
  }
  return status === filter;
}

/**
 * The review queue.
 *
 * One example at a time by default, because approving is a judgement per
 * example. Keyboard controls are provided for the common case of working
 * through a backlog in one sitting.
 */
export function ReviewExamplesPage() {
  const courseId = useCourseId();
  const {
    examples,
    counts,
    loading,
    error,
    busyId,
    actionMessage,
    actionFailed,
    reload,
    approve,
    reject,
    saveEdit,
  } = useExampleReview(courseId);

  const [filter, setFilter] = useState<ReviewFilter>('pending');
  const [index, setIndex] = useState(0);

  const visible = useMemo(
    () => examples.filter((example) => matchesFilter(resolveExampleStatus(example), filter)),
    [examples, filter],
  );

  // Approving the last item in a filter would otherwise leave the queue past
  // its end.
  useEffect(() => {
    setIndex((current) => Math.min(current, Math.max(0, visible.length - 1)));
  }, [visible.length]);

  useEffect(() => {
    setIndex(0);
  }, [filter]);

  const current = visible[index];
  const busy = busyId !== null;

  const goPrevious = useCallback(() => {
    setIndex((value) => Math.max(0, value - 1));
  }, []);

  const goNext = useCallback(() => {
    setIndex((value) => Math.min(visible.length - 1, value + 1));
  }, [visible.length]);

  // Keyboard shortcuts for working through a backlog. Ignored while typing so
  // they never fire inside the edit form.
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (
        busy ||
        !current ||
        (target &&
          (target.tagName === 'INPUT' ||
            target.tagName === 'TEXTAREA' ||
            target.isContentEditable))
      ) {
        return;
      }

      if (event.key === 'ArrowLeft' || event.key === 'k') {
        goPrevious();
      } else if (event.key === 'ArrowRight' || event.key === 'j') {
        goNext();
      } else if (event.key === 'a') {
        void approve(String(current.id || ''));
      } else if (event.key === 'r') {
        void reject(String(current.id || ''));
      }
    }

    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [approve, busy, current, goNext, goPrevious, reject]);

  const handleSaveEdit = useCallback(
    (seedId: string, draft: ReviewDraft) => {
      void saveEdit(seedId, draft);
    },
    [saveEdit],
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
      </div>

      {loading ? (
        <p className="ui-text-muted" role="status" aria-live="polite">
          Loading examples…
        </p>
      ) : visible.length === 0 ? (
        <EmptyState
          illustration="empty-course"
          title={filter === 'pending' ? 'Nothing waiting for you' : 'Nothing to show'}
          description={
            counts.total === 0
              ? 'No example questions have been collected for this course yet.'
              : filter === 'pending'
                ? 'Every example has been reviewed. Switch filters to revisit your decisions.'
                : 'No examples match this filter.'
          }
        />
      ) : (
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
              key={String(current.id || index)}
              example={current}
              busy={busyId === String(current.id || '').trim()}
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
      )}
        </>
      )}
    </div>
  );
}
