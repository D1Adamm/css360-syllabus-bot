import { useCallback, useEffect, useMemo, useState } from 'react';
import { PageHeader } from '../components/PageHeader';
import { ReviewSeedCard } from '../components/ReviewSeedCard';
import { useCourseId } from '../context/CourseContext';
import {
  ApiError,
  type CourseSeedReviewRecord,
  type SeedReviewStatus,
  exportApprovedCourseSeeds,
  getApprovedExportStatus,
  listCourseSeeds,
  prepareTrainingSplit,
  reviewCourseSeed,
} from '../lib/api';

type ReviewFilter = 'all' | SeedReviewStatus;

const FILTERS: { id: ReviewFilter; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'generated', label: 'Generated' },
  { id: 'approved', label: 'Approved' },
  { id: 'rejected', label: 'Rejected' },
  { id: 'edited', label: 'Edited' },
];

function resolveStatus(seed: CourseSeedReviewRecord): SeedReviewStatus | string {
  return String(seed.reviewStatus || seed.status || 'generated')
    .trim()
    .toLowerCase();
}

function seedWasEdited(seed: CourseSeedReviewRecord): boolean {
  if (seed.wasEdited === true) {
    return true;
  }
  if (resolveStatus(seed) === 'edited') {
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

function questionOf(seed: CourseSeedReviewRecord): string {
  return String(seed.question || seed.instruction || '').trim();
}

function answerOf(seed: CourseSeedReviewRecord): string {
  return String(seed.answer || seed.response || '').trim();
}

function countByStatus(seeds: CourseSeedReviewRecord[]): Record<string, number> {
  const counts: Record<string, number> = {
    generated: 0,
    approved: 0,
    rejected: 0,
    edited: 0,
  };
  for (const seed of seeds) {
    const status = resolveStatus(seed);
    counts[status] = (counts[status] || 0) + 1;
  }
  return counts;
}

function patchSeedLocally(
  current: CourseSeedReviewRecord[],
  seedId: string,
  patch: Partial<CourseSeedReviewRecord> & { reviewStatus: SeedReviewStatus },
): CourseSeedReviewRecord[] {
  return current.map((seed) => {
    if (String(seed.id || '').trim() !== seedId) {
      return seed;
    }
    const next: CourseSeedReviewRecord = {
      ...seed,
      ...patch,
      id: seedId,
      // Keep both fields in sync so filters/counts update immediately.
      reviewStatus: patch.reviewStatus,
      status: patch.reviewStatus,
    };
    if (patch.wasEdited === true || seed.wasEdited === true) {
      next.wasEdited = true;
    }
    if (next.originalQuestion === undefined && seed.originalQuestion) {
      next.originalQuestion = seed.originalQuestion;
    }
    if (next.originalAnswer === undefined && seed.originalAnswer) {
      next.originalAnswer = seed.originalAnswer;
    }
    return next;
  });
}

function mergeReviewedSeed(
  current: CourseSeedReviewRecord[],
  seedId: string,
  updated: CourseSeedReviewRecord,
  fallbackStatus: SeedReviewStatus,
): CourseSeedReviewRecord[] {
  const status = (resolveStatus({
    ...updated,
    reviewStatus: updated.reviewStatus || fallbackStatus,
  }) || fallbackStatus) as SeedReviewStatus;
  return current.map((seed) => {
    if (String(seed.id || '').trim() !== seedId) {
      return seed;
    }
    const merged: CourseSeedReviewRecord = {
      ...seed,
      ...updated,
      id: seedId,
      reviewStatus: status,
      status,
    };
    if (seedWasEdited(seed) || seedWasEdited(merged)) {
      merged.wasEdited = true;
    }
    if (!merged.originalQuestion && seed.originalQuestion) {
      merged.originalQuestion = seed.originalQuestion;
    }
    if (!merged.originalAnswer && seed.originalAnswer) {
      merged.originalAnswer = seed.originalAnswer;
    }
    return merged;
  });
}

export function ReviewSeedsPage() {
  const courseId = useCourseId();
  const [seeds, setSeeds] = useState<CourseSeedReviewRecord[]>([]);
  const [firebasePath, setFirebasePath] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<ReviewFilter>('all');
  const [busySeedId, setBusySeedId] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [preparingSplit, setPreparingSplit] = useState(false);
  const [hasApprovedExport, setHasApprovedExport] = useState(false);

  const refreshApprovedExportStatus = useCallback(async () => {
    try {
      const status = await getApprovedExportStatus(courseId);
      setHasApprovedExport(Boolean(status.exists));
    } catch {
      setHasApprovedExport(false);
    }
  }, [courseId]);

  const loadSeeds = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await listCourseSeeds(courseId);
      setSeeds(response.seeds || []);
      setFirebasePath(response.firebasePath || `courses/${courseId}/seedExamples`);
      await refreshApprovedExportStatus();
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : 'Could not load seeds for review.';
      setError(message);
      setSeeds([]);
      setHasApprovedExport(false);
    } finally {
      setLoading(false);
    }
  }, [courseId, refreshApprovedExportStatus]);

  useEffect(() => {
    void loadSeeds();
  }, [loadSeeds]);

  const counts = useMemo(() => countByStatus(seeds), [seeds]);

  const filteredSeeds = useMemo(() => {
    if (filter === 'all') {
      return seeds;
    }
    return seeds.filter((seed) => resolveStatus(seed) === filter);
  }, [seeds, filter]);

  async function applyReviewAction(
    seedId: string,
    reviewStatus: SeedReviewStatus,
    body: {
      reviewStatus: SeedReviewStatus;
      question?: string;
      answer?: string;
      reviewNotes?: string;
    },
    successMessage: string,
  ) {
    setBusySeedId(seedId);
    setActionMessage(null);

    let snapshot: CourseSeedReviewRecord[] = [];
    // Optimistic update so the card leaves the current filter list immediately.
    setSeeds((current) => {
      snapshot = current;
      const existing = current.find((seed) => String(seed.id || '').trim() === seedId);
      const markingEdited =
        reviewStatus === 'edited' ||
        body.question !== undefined ||
        body.answer !== undefined;
      const wasEdited =
        Boolean(existing && seedWasEdited(existing)) || markingEdited;
      const patch: Partial<CourseSeedReviewRecord> & {
        reviewStatus: SeedReviewStatus;
      } = {
        ...(body.question !== undefined
          ? { question: body.question, instruction: body.question }
          : {}),
        ...(body.answer !== undefined
          ? { answer: body.answer, response: body.answer }
          : {}),
        ...(body.reviewNotes !== undefined ? { reviewNotes: body.reviewNotes } : {}),
        reviewStatus,
      };
      if (wasEdited) {
        patch.wasEdited = true;
      }
      if (markingEdited && existing) {
        if (!String(existing.originalQuestion || '').trim()) {
          patch.originalQuestion = questionOf(existing);
        }
        if (!String(existing.originalAnswer || '').trim()) {
          patch.originalAnswer = answerOf(existing);
        }
      }
      return patchSeedLocally(current, seedId, patch);
    });
    try {
      const response = await reviewCourseSeed(courseId, seedId, body);
      setSeeds((current) =>
        mergeReviewedSeed(current, seedId, response.seed, reviewStatus),
      );
      setActionMessage(successMessage);
    } catch (err) {
      setSeeds(snapshot);
      setActionMessage(
        err instanceof ApiError ? err.message : `Could not mark seed as ${reviewStatus}.`,
      );
    } finally {
      setBusySeedId(null);
    }
  }

  async function handleApprove(seedId: string) {
    await applyReviewAction(
      seedId,
      'approved',
      { reviewStatus: 'approved' },
      'Seed approved.',
    );
  }

  async function handleReject(seedId: string) {
    await applyReviewAction(
      seedId,
      'rejected',
      { reviewStatus: 'rejected' },
      'Seed rejected.',
    );
  }

  async function handleEditSave(
    seedId: string,
    draft: { question: string; answer: string; reviewNotes?: string },
  ) {
    await applyReviewAction(
      seedId,
      'edited',
      {
        reviewStatus: 'edited',
        question: draft.question,
        answer: draft.answer,
        reviewNotes: draft.reviewNotes,
      },
      'Seed edited. Provenance metadata was preserved.',
    );
  }

  async function handleExportApproved() {
    setExporting(true);
    setActionMessage(null);
    try {
      const response = await exportApprovedCourseSeeds(courseId);
      const count =
        Number(response.summary?.validatedCount) ||
        Number(response.summary?.exportedCount) ||
        Number(response.summary?.approvedCount) ||
        0;
      const path =
        String(response.summary?.exportPath || response.summary?.files?.finetuneJsonl || '').trim();
      const arrowPath = path ? ` → ${path}` : '';
      setHasApprovedExport(true);
      setActionMessage(
        `Exported and validated ${count} approved seed${count === 1 ? '' : 's'}${arrowPath}`,
      );
    } catch (err) {
      setActionMessage(
        err instanceof ApiError ? err.message : 'Export approved failed.',
      );
    } finally {
      setExporting(false);
    }
  }

  async function handlePrepareTrainingSplit() {
    setPreparingSplit(true);
    setActionMessage(null);
    try {
      const response = await prepareTrainingSplit(courseId);
      const train = Number(response.summary?.trainExamples) || 0;
      const validation = Number(response.summary?.validationExamples) || 0;
      setActionMessage(
        `Prepared training split: ${train} train, ${validation} validation`,
      );
    } catch (err) {
      setActionMessage(
        err instanceof ApiError ? err.message : 'Prepare training split failed.',
      );
    } finally {
      setPreparingSplit(false);
    }
  }

  return (
    <>
      <PageHeader
        title="Review Seeds"
        description="Approve, reject, or edit AI-generated starter seeds for this course. Validated seeds stay generated until you review them."
      />

      <aside className="dataset-notice" aria-label="Review notice">
        <p>
          <strong>Course-isolated review:</strong> only seeds from{' '}
          <code>{firebasePath || `courses/${courseId}/seedExamples`}</code>. Seeds are
          not auto-approved.
        </p>
      </aside>

      <div className="review-toolbar">
        <div className="review-counts" aria-label="Review status counts">
          <span>
            Generated: <strong>{counts.generated || 0}</strong>
          </span>
          <span>
            Approved: <strong>{counts.approved || 0}</strong>
          </span>
          <span>
            Rejected: <strong>{counts.rejected || 0}</strong>
          </span>
          <span>
            Edited: <strong>{counts.edited || 0}</strong>
          </span>
          <span>
            All: <strong>{seeds.length}</strong>
          </span>
        </div>

        <div className="review-toolbar__actions">
          <button
            type="button"
            className="review-toolbar__button"
            onClick={() => void loadSeeds()}
            disabled={loading || busySeedId !== null || exporting || preparingSplit}
          >
            Refresh
          </button>
          <button
            type="button"
            className="review-toolbar__button review-toolbar__button--primary"
            onClick={() => void handleExportApproved()}
            disabled={loading || exporting || preparingSplit || busySeedId !== null}
          >
            {exporting ? 'Exporting…' : 'Export Approved'}
          </button>
          <button
            type="button"
            className="review-toolbar__button"
            onClick={() => void handlePrepareTrainingSplit()}
            disabled={
              loading ||
              exporting ||
              preparingSplit ||
              busySeedId !== null ||
              !hasApprovedExport
            }
            title={
              hasApprovedExport
                ? 'Create deterministic train/validation files from the approved export'
                : 'Export approved seeds first'
            }
          >
            {preparingSplit ? 'Preparing split…' : 'Prepare Training Split'}
          </button>
        </div>
      </div>

      <div className="review-filters" role="tablist" aria-label="Filter by review status">
        {FILTERS.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={filter === item.id}
            className={
              filter === item.id
                ? 'review-filters__button review-filters__button--active'
                : 'review-filters__button'
            }
            onClick={() => setFilter(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>

      {actionMessage && (
        <p className="seed-builder-status" role="status" aria-live="polite">
          {actionMessage}
        </p>
      )}

      {error && (
        <p className="seed-builder-status seed-builder-status--error" role="alert">
          {error}
        </p>
      )}

      {loading ? (
        <p className="seed-builder-status" role="status" aria-live="polite">
          Loading seeds for review…
        </p>
      ) : filteredSeeds.length === 0 ? (
        <section className="dataset-empty" aria-live="polite">
          <h2 className="dataset-empty__title">No seeds to show</h2>
          <p className="dataset-empty__text">
            {seeds.length === 0
              ? 'No seed examples are saved for this course yet.'
              : 'No seeds match this review filter.'}
          </p>
        </section>
      ) : (
        <div className="review-seed-list">
          {filteredSeeds.map((seed, index) => {
            const id = String(seed.id || '');
            return (
              <ReviewSeedCard
                key={id || questionTextFallback(seed)}
                seed={seed}
                displayNumber={index + 1}
                busy={busySeedId === id}
                onApprove={handleApprove}
                onReject={handleReject}
                onEditSave={handleEditSave}
              />
            );
          })}
        </div>
      )}
    </>
  );
}

function questionTextFallback(seed: CourseSeedReviewRecord): string {
  return String(seed.question || seed.instruction || Math.random());
}
