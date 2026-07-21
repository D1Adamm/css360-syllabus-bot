import { useId, useState } from 'react';
import type { CourseSeedReviewRecord, SeedReviewStatus } from '../lib/api';

export interface ReviewSeedCardProps {
  seed: CourseSeedReviewRecord;
  displayNumber: number;
  busy?: boolean;
  onApprove: (seedId: string) => void;
  onReject: (seedId: string) => void;
  onEditSave: (
    seedId: string,
    draft: { question: string; answer: string; reviewNotes?: string },
  ) => void;
}

function resolveReviewStatus(seed: CourseSeedReviewRecord): SeedReviewStatus | string {
  const raw = (seed.reviewStatus || seed.status || 'generated').trim().toLowerCase();
  if (
    raw === 'generated' ||
    raw === 'approved' ||
    raw === 'rejected' ||
    raw === 'edited'
  ) {
    return raw;
  }
  return raw || 'generated';
}

function seedWasEdited(seed: CourseSeedReviewRecord): boolean {
  if (seed.wasEdited === true) {
    return true;
  }
  if (resolveReviewStatus(seed) === 'edited') {
    return true;
  }
  if (String(seed.originalQuestion || '').trim()) {
    return true;
  }
  if (String(seed.originalAnswer || '').trim()) {
    return true;
  }
  return false;
}

function questionText(seed: CourseSeedReviewRecord): string {
  return String(seed.question || seed.instruction || '').trim();
}

function answerText(seed: CourseSeedReviewRecord): string {
  return String(seed.answer || seed.response || '').trim();
}

export function ReviewSeedCard({
  seed,
  displayNumber,
  busy = false,
  onApprove,
  onReject,
  onEditSave,
}: ReviewSeedCardProps) {
  const seedId = String(seed.id || '').trim();
  const status = resolveReviewStatus(seed);
  const wasEdited = seedWasEdited(seed);
  const [editing, setEditing] = useState(false);
  const [draftQuestion, setDraftQuestion] = useState(questionText(seed));
  const [draftAnswer, setDraftAnswer] = useState(answerText(seed));
  const [draftNotes, setDraftNotes] = useState(seed.reviewNotes || '');
  const editFormId = useId();

  function startEdit() {
    setDraftQuestion(questionText(seed));
    setDraftAnswer(answerText(seed));
    setDraftNotes(seed.reviewNotes || '');
    setEditing(true);
  }

  function cancelEdit() {
    setEditing(false);
    setDraftQuestion(questionText(seed));
    setDraftAnswer(answerText(seed));
    setDraftNotes(seed.reviewNotes || '');
  }

  function saveEdit() {
    if (!seedId) {
      return;
    }
    onEditSave(seedId, {
      question: draftQuestion.trim(),
      answer: draftAnswer.trim(),
      reviewNotes: draftNotes.trim() || undefined,
    });
    setEditing(false);
  }

  const score =
    typeof seed.validation?.score === 'number' ? seed.validation.score : null;
  const evidence = String(seed.evidenceQuote || seed.sourceSection || '').trim();

  return (
    <article className="review-seed-card" data-review-status={status}>
      <div className="review-seed-card__top">
        <span className="review-seed-card__number" aria-label={`Seed number ${displayNumber}`}>
          Seed #{displayNumber}
        </span>
        <div className="review-seed-card__labels">
          <span className="review-seed-card__category">
            {seed.category?.trim() || 'Uncategorized'}
          </span>
          <span className={`review-seed-card__status review-seed-card__status--${status}`}>
            {status}
          </span>
          {wasEdited && status !== 'edited' && (
            <span className="review-seed-card__status review-seed-card__status--edited">
              Edited
            </span>
          )}
          {score !== null && (
            <span className="review-seed-card__score">
              Validation {Math.round(score * 100)}%
            </span>
          )}
          {seed.origin && (
            <span className="review-seed-card__origin">{seed.origin}</span>
          )}
        </div>
      </div>

      {editing ? (
        <form
          id={editFormId}
          className="review-seed-card__edit-form"
          onSubmit={(event) => {
            event.preventDefault();
            saveEdit();
          }}
        >
          <label className="review-seed-card__field">
            <span>Question</span>
            <textarea
              value={draftQuestion}
              onChange={(event) => setDraftQuestion(event.target.value)}
              rows={3}
              required
              disabled={busy}
            />
          </label>
          <label className="review-seed-card__field">
            <span>Answer</span>
            <textarea
              value={draftAnswer}
              onChange={(event) => setDraftAnswer(event.target.value)}
              rows={5}
              required
              disabled={busy}
            />
          </label>
          <label className="review-seed-card__field">
            <span>Review notes (optional)</span>
            <input
              type="text"
              value={draftNotes}
              onChange={(event) => setDraftNotes(event.target.value)}
              disabled={busy}
            />
          </label>
        </form>
      ) : (
        <>
          <h2 className="review-seed-card__question">{questionText(seed)}</h2>
          <p className="review-seed-card__answer">{answerText(seed)}</p>
        </>
      )}

      <div className="review-seed-card__meta">
        {evidence && (
          <p>
            <span className="review-seed-card__meta-label">Evidence / source:</span>{' '}
            {evidence}
          </p>
        )}
        {seed.factId && (
          <p>
            <span className="review-seed-card__meta-label">Fact ID:</span> {seed.factId}
          </p>
        )}
        {seed.validation?.reason && (
          <p>
            <span className="review-seed-card__meta-label">Validation reason:</span>{' '}
            {seed.validation.reason}
          </p>
        )}
        {seed.originalQuestion && (
          <p>
            <span className="review-seed-card__meta-label">Original question:</span>{' '}
            {seed.originalQuestion}
          </p>
        )}
        {seed.originalAnswer && (
          <p>
            <span className="review-seed-card__meta-label">Original answer:</span>{' '}
            {seed.originalAnswer}
          </p>
        )}
        {seedId && (
          <p>
            <span className="review-seed-card__meta-label">ID:</span> {seedId}
          </p>
        )}
      </div>

      <div className="review-seed-card__actions">
        {editing ? (
          <>
            <button
              type="submit"
              form={editFormId}
              className="review-seed-card__button review-seed-card__button--primary"
              disabled={busy || !draftQuestion.trim() || !draftAnswer.trim()}
            >
              Save edit
            </button>
            <button
              type="button"
              className="review-seed-card__button"
              onClick={cancelEdit}
              disabled={busy}
            >
              Cancel
            </button>
          </>
        ) : (
          <>
            <button
              type="button"
              className="review-seed-card__button review-seed-card__button--approve"
              onClick={() => seedId && onApprove(seedId)}
              disabled={busy || !seedId || status === 'approved'}
            >
              Approve
            </button>
            <button
              type="button"
              className="review-seed-card__button review-seed-card__button--reject"
              onClick={() => seedId && onReject(seedId)}
              disabled={busy || !seedId || status === 'rejected'}
            >
              Reject
            </button>
            <button
              type="button"
              className="review-seed-card__button"
              onClick={startEdit}
              disabled={busy || !seedId}
            >
              Edit
            </button>
          </>
        )}
      </div>
    </article>
  );
}
