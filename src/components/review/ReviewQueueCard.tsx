import { useEffect, useId, useState } from 'react';
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

export interface ReviewQueueCardProps {
  example: CourseSeedReviewRecord;
  busy: boolean;
  onApprove: (seedId: string) => void;
  onReject: (seedId: string) => void;
  onSaveEdit: (seedId: string, draft: ReviewDraft) => void;
}

const STATUS_LABEL: Record<string, string> = {
  generated: 'Awaiting review',
  approved: 'Approved',
  rejected: 'Rejected',
  edited: 'Edited',
};

function sourceLabel(example: CourseSeedReviewRecord): string {
  return String(example.origin || '').trim() === 'user'
    ? 'Student submitted'
    : 'AI generated';
}

/**
 * One example, reviewed on its own.
 *
 * A queue rather than a spreadsheet: reviewing is a judgement call per example,
 * and a table encourages skimming rows instead of reading answers. The
 * evidence quote sits directly under the answer because "is this actually what
 * the syllabus says" is the question being answered.
 */
export function ReviewQueueCard({
  example,
  busy,
  onApprove,
  onReject,
  onSaveEdit,
}: ReviewQueueCardProps) {
  const fieldId = useId();
  const seedId = String(example.id || '').trim();
  const status = resolveExampleStatus(example);
  const wasEdited = exampleWasEdited(example);

  const [editing, setEditing] = useState(false);
  const [question, setQuestion] = useState(exampleQuestion(example));
  const [answer, setAnswer] = useState(exampleAnswer(example));
  const [notes, setNotes] = useState(example.reviewNotes || '');

  // Moving to another example must not carry the previous draft across.
  useEffect(() => {
    setEditing(false);
    setQuestion(exampleQuestion(example));
    setAnswer(exampleAnswer(example));
    setNotes(example.reviewNotes || '');
  }, [example]);

  const evidence = String(example.evidenceQuote || '').trim();
  const section = String(example.sourceSection || '').trim();
  const category = String(example.category || '').trim();

  function save() {
    if (!seedId) {
      return;
    }
    onSaveEdit(seedId, {
      question: question.trim(),
      answer: answer.trim(),
      reviewNotes: notes.trim() || undefined,
    });
  }

  return (
    <article className="review-card">
      <header className="review-card__header">
        <div className="review-card__labels">
          <StatusPill
            tone={
              status === 'approved'
                ? 'success'
                : status === 'rejected'
                  ? 'danger'
                  : status === 'edited'
                    ? 'info'
                    : 'warning'
            }
          >
            {STATUS_LABEL[status] ?? 'Awaiting review'}
          </StatusPill>
          <span className="review-card__source">{sourceLabel(example)}</span>
          {category && <span className="review-card__category">{category}</span>}
          {wasEdited && status !== 'edited' && (
            <StatusPill tone="neutral" dot={false}>
              Edited
            </StatusPill>
          )}
        </div>
      </header>

      {editing ? (
        <div className="review-card__edit">
          <label className="ui-field__label" htmlFor={`${fieldId}-question`}>
            Question
          </label>
          <textarea
            id={`${fieldId}-question`}
            className="review-card__input"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            rows={2}
            disabled={busy}
          />

          <label className="ui-field__label" htmlFor={`${fieldId}-answer`}>
            Answer
          </label>
          <textarea
            id={`${fieldId}-answer`}
            className="review-card__input"
            value={answer}
            onChange={(event) => setAnswer(event.target.value)}
            rows={5}
            disabled={busy}
          />

          <label className="ui-field__label" htmlFor={`${fieldId}-notes`}>
            Note to yourself <span className="ui-field__requirement">(optional)</span>
          </label>
          <input
            id={`${fieldId}-notes`}
            className="review-card__input"
            value={notes}
            onChange={(event) => setNotes(event.target.value)}
            disabled={busy}
          />

          <div className="review-card__actions">
            <Button variant="tertiary" onClick={() => setEditing(false)} disabled={busy}>
              Cancel
            </Button>
            <Button
              variant="primary"
              onClick={save}
              loading={busy}
              loadingLabel="Saving…"
              disabled={!question.trim() || !answer.trim()}
            >
              Save changes
            </Button>
          </div>
        </div>
      ) : (
        <>
          <h2 className="review-card__question">{exampleQuestion(example)}</h2>
          <p className="review-card__answer">{exampleAnswer(example)}</p>

          {evidence && (
            <figure className="review-card__evidence">
              <blockquote>{evidence}</blockquote>
              {section && <figcaption>{section}</figcaption>}
            </figure>
          )}

          {!evidence && section && (
            <p className="review-card__section">From: {section}</p>
          )}

          {wasEdited && example.originalQuestion && (
            <details className="review-card__history">
              <summary>What this said before you edited it</summary>
              <p className="review-card__history-question">
                {example.originalQuestion}
              </p>
              {example.originalAnswer && (
                <p className="review-card__history-answer">{example.originalAnswer}</p>
              )}
            </details>
          )}

          <div className="review-card__actions">
            <Button
              variant="tertiary"
              onClick={() => onReject(seedId)}
              disabled={busy}
            >
              Reject
            </Button>
            <Button variant="secondary" onClick={() => setEditing(true)} disabled={busy}>
              Edit
            </Button>
            <Button
              variant="primary"
              onClick={() => onApprove(seedId)}
              loading={busy}
              loadingLabel="Saving…"
            >
              Approve
            </Button>
          </div>
        </>
      )}
    </article>
  );
}
