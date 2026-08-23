import { useEffect, useId, useRef } from 'react';
import { Button } from '../ui/Button';
import { StatusPill } from '../ui/StatusPill';
import type { CourseSeedReviewRecord } from '../../lib/api';
import {
  exampleAnswer,
  exampleQuestion,
  exampleWasEdited,
  resolveExampleStatus,
} from '../../lib/exampleCounts';
import type { ReviewDraft } from '../../hooks/useExampleReview';
import {
  exampleSourceLabel,
  reviewStatusLabel,
  reviewStatusTone,
} from './reviewLabels';

export interface ReviewListRowProps {
  example: CourseSeedReviewRecord;
  /** 1-based position within the current filter. */
  position: number;
  total: number;
  expanded: boolean;
  selected: boolean;
  /** The live draft when this row is being edited, otherwise null. */
  draft: ReviewDraft | null;
  /** The row the keyboard shortcuts currently act on. */
  focused: boolean;
  /** This row has a write in flight. */
  busy: boolean;
  /** Something else is writing; this row's actions must not queue behind it. */
  disabled: boolean;
  duplicate: boolean;
  onToggleExpanded: (seedId: string) => void;
  onToggleSelected: (seedId: string) => void;
  onFocusRow: (seedId: string) => void;
  onStartEdit: (seedId: string) => void;
  onDraftChange: (patch: Partial<ReviewDraft>) => void;
  onCancelEdit: () => void;
  onSaveEdit: (seedId: string, draft: ReviewDraft) => void;
  onApprove: (seedId: string) => void;
  onReject: (seedId: string) => void;
}

/**
 * One example as a row in the fast review list.
 *
 * The whole point of this surface is a professor working through fifty of
 * these in one sitting, so every decision they can make is on the row itself:
 * approve, reject, and edit are visible without opening anything, and editing
 * happens in place rather than in a dialog that would cost two clicks and the
 * reader's place in the list.
 *
 * Collapsed still shows the question, the status and the actions — a row you
 * cannot act on without expanding it is a row that has not saved anyone any
 * time. It also shows the answer in full: the answer is the thing being
 * approved, and a professor cannot judge a sentence they can only see the
 * first line of. Collapsing hides the supporting material — the syllabus
 * evidence and the edit history — not the decision itself.
 */
export function ReviewListRow({
  example,
  position,
  total,
  expanded,
  selected,
  draft,
  focused,
  busy,
  disabled,
  duplicate,
  onToggleExpanded,
  onToggleSelected,
  onFocusRow,
  onStartEdit,
  onDraftChange,
  onCancelEdit,
  onSaveEdit,
  onApprove,
  onReject,
}: ReviewListRowProps) {
  const fieldId = useId();
  const rowRef = useRef<HTMLLIElement>(null);
  const seedId = String(example.id || '').trim();
  const status = resolveExampleStatus(example);
  const wasEdited = exampleWasEdited(example);

  const question = exampleQuestion(example);
  const answer = exampleAnswer(example);

  /*
   * The draft lives on the page, not here.
   *
   * A save optimistically moves the example to `edited`, which under the
   * "Awaiting review" filter takes this row out of the list and unmounts it.
   * If the save then fails, the row comes back — and local draft state would
   * have died with it, losing the professor's wording at the exact moment the
   * error message tells them to try again.
   */

  // Keyboard navigation moves a highlight, so the highlighted row has to come
  // into view on its own. Guarded because jsdom has no scrollIntoView.
  useEffect(() => {
    if (!focused || typeof rowRef.current?.scrollIntoView !== 'function') {
      return;
    }
    rowRef.current.scrollIntoView({ block: 'nearest' });
  }, [focused]);

  const evidence = String(example.evidenceQuote || '').trim();
  const section = String(example.sourceSection || '').trim();
  const category = String(example.category || '').trim();
  const blocked = busy || disabled;

  function save() {
    if (!seedId || !draft) {
      return;
    }
    onSaveEdit(seedId, {
      question: draft.question.trim(),
      answer: draft.answer.trim(),
    });
  }

  return (
    <li
      ref={rowRef}
      className={[
        'review-row',
        focused ? 'review-row--focused' : '',
        selected ? 'review-row--selected' : '',
        expanded ? 'review-row--expanded' : '',
      ]
        .filter(Boolean)
        .join(' ')}
      aria-label={`Example ${position} of ${total}`}
      onFocusCapture={() => onFocusRow(seedId)}
    >
      <div className="review-row__bar">
        <input
          type="checkbox"
          id={`${fieldId}-select`}
          className="review-row__checkbox"
          checked={selected}
          disabled={disabled}
          onChange={() => onToggleSelected(seedId)}
        />
        <label className="ui-visually-hidden" htmlFor={`${fieldId}-select`}>
          Select example {position}
        </label>

        <span className="review-row__position">
          {position} of {total}
        </span>

        <StatusPill tone={reviewStatusTone(status)}>
          {reviewStatusLabel(status)}
        </StatusPill>

        {wasEdited && status !== 'edited' && (
          <StatusPill tone="neutral" dot={false}>
            Edited
          </StatusPill>
        )}

        {duplicate && (
          <span className="review-row__duplicate">Possible duplicate</span>
        )}

        <span className="review-row__meta">
          {[exampleSourceLabel(example), category, section]
            .filter(Boolean)
            .join(' · ')}
        </span>

        <Button
          size="sm"
          variant="ghost"
          className="review-row__disclosure"
          aria-expanded={expanded}
          aria-label={`${expanded ? 'Collapse' : 'Expand'} example ${position}`}
          onClick={() => onToggleExpanded(seedId)}
        >
          {expanded ? 'Collapse' : 'Expand'}
        </Button>
      </div>

      {draft ? (
        <div className="review-row__edit">
          <label className="ui-field__label" htmlFor={`${fieldId}-question`}>
            Question
          </label>
          <textarea
            id={`${fieldId}-question`}
            className="review-row__input"
            value={draft.question}
            onChange={(event) => onDraftChange({ question: event.target.value })}
            rows={2}
            disabled={busy}
          />

          <label className="ui-field__label" htmlFor={`${fieldId}-answer`}>
            Answer
          </label>
          <textarea
            id={`${fieldId}-answer`}
            className="review-row__input"
            value={draft.answer}
            onChange={(event) => onDraftChange({ answer: event.target.value })}
            rows={5}
            disabled={busy}
          />

          <div className="review-row__actions">
            <Button
              size="sm"
              variant="tertiary"
              onClick={onCancelEdit}
              disabled={busy}
              aria-label={`Cancel editing example ${position}`}
            >
              Cancel
            </Button>
            <Button
              size="sm"
              variant="primary"
              onClick={save}
              loading={busy}
              loadingLabel="Saving…"
              disabled={!draft.question.trim() || !draft.answer.trim()}
              aria-label={`Save changes to example ${position}`}
            >
              Save changes
            </Button>
          </div>
        </div>
      ) : (
        <>
          <p className="review-row__question">{question}</p>

          <p className="review-row__answer">{answer}</p>

          {expanded && evidence && (
            <figure className="review-row__evidence">
              <blockquote>{evidence}</blockquote>
              {section && <figcaption>{section}</figcaption>}
            </figure>
          )}

          {expanded && wasEdited && example.originalQuestion && (
            <details className="review-row__history">
              <summary>What this said before you edited it</summary>
              <p className="review-row__history-question">
                {example.originalQuestion}
              </p>
              {example.originalAnswer && (
                <p className="review-row__history-answer">{example.originalAnswer}</p>
              )}
            </details>
          )}

          <div className="review-row__actions">
            <Button
              size="sm"
              variant="tertiary"
              onClick={() => onReject(seedId)}
              disabled={blocked}
              aria-label={`Reject example ${position}`}
            >
              Reject
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => onStartEdit(seedId)}
              disabled={blocked}
              aria-label={`Edit example ${position}`}
            >
              Edit
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => onApprove(seedId)}
              loading={busy}
              loadingLabel="Saving…"
              disabled={disabled}
              aria-label={`Approve example ${position}`}
            >
              Approve
            </Button>
          </div>
        </>
      )}
    </li>
  );
}
