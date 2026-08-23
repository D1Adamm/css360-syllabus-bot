import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ApiError,
  type CourseSeedReviewRecord,
  type SeedReviewStatus,
  listCourseSeeds,
  reviewCourseSeed,
} from '../lib/api';
import {
  countExamples,
  exampleAnswer,
  exampleQuestion,
  exampleWasEdited,
  resolveExampleStatus,
  type ExampleCounts,
} from '../lib/exampleCounts';

/**
 * Loading and reviewing one course's examples.
 *
 * Lifted out of the review page unchanged. The optimistic update, the snapshot
 * rollback on failure, and the provenance rules are the delicate parts and are
 * carried over verbatim:
 *
 *   - `wasEdited` is sticky. Once an example has been edited it stays marked,
 *     including after a later approval.
 *   - `originalQuestion` / `originalAnswer` are captured on the first edit only,
 *     so the first human change is preserved rather than the most recent one.
 *   - A failed request restores the exact pre-request list.
 */

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
  const status = (resolveExampleStatus({
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
    if (exampleWasEdited(seed) || exampleWasEdited(merged)) {
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

export interface ReviewDraft {
  question: string;
  answer: string;
  reviewNotes?: string;
}

/** Outcome of a bulk review run. Failures are reported, never hidden. */
export interface BulkReviewResult {
  succeeded: string[];
  failed: string[];
}

/**
 * How many review requests are in flight at once during a bulk action.
 *
 * Selecting fifty examples must not become fifty simultaneous requests: the
 * backend writes each one to Postgres, and a burst is how a review session
 * turns into a pile of timeouts. Four keeps a batch of fifty comfortably
 * quick while staying polite.
 */
export const BULK_REVIEW_CONCURRENCY = 4;

export interface UseExampleReviewResult {
  examples: CourseSeedReviewRecord[];
  counts: ExampleCounts;
  loading: boolean;
  error: string | null;
  busyId: string | null;
  /** True while a bulk action is running. */
  bulkBusy: boolean;
  actionMessage: string | null;
  actionFailed: boolean;
  reload: () => Promise<void>;
  /** Resolves true when the write landed; false when it failed and rolled back. */
  approve: (seedId: string) => Promise<boolean>;
  reject: (seedId: string) => Promise<boolean>;
  saveEdit: (seedId: string, draft: ReviewDraft) => Promise<boolean>;
  reviewMany: (
    seedIds: string[],
    reviewStatus: Extract<SeedReviewStatus, 'approved' | 'rejected'>,
  ) => Promise<BulkReviewResult>;
}

export function useExampleReview(courseId: string): UseExampleReviewResult {
  const [examples, setExamples] = useState<CourseSeedReviewRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [bulkBusy, setBulkBusy] = useState(false);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [actionFailed, setActionFailed] = useState(false);

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await listCourseSeeds(courseId);
      setExamples(response.seeds || []);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : 'These examples could not be loaded. Try again in a moment.',
      );
      setExamples([]);
    } finally {
      setLoading(false);
    }
  }, [courseId]);

  useEffect(() => {
    void reload();
  }, [reload]);

  const counts = useMemo(() => countExamples(examples), [examples]);

  const applyReviewAction = useCallback(
    async (
      seedId: string,
      reviewStatus: SeedReviewStatus,
      body: {
        reviewStatus: SeedReviewStatus;
        question?: string;
        answer?: string;
        reviewNotes?: string;
      },
      successMessage: string,
    ): Promise<boolean> => {
      setBusyId(seedId);
      setActionMessage(null);
      setActionFailed(false);

      let snapshot: CourseSeedReviewRecord[] = [];
      // Optimistic update so the card leaves the queue immediately.
      setExamples((current) => {
        snapshot = current;
        const existing = current.find(
          (seed) => String(seed.id || '').trim() === seedId,
        );
        const markingEdited =
          reviewStatus === 'edited' ||
          body.question !== undefined ||
          body.answer !== undefined;
        const wasEdited =
          Boolean(existing && exampleWasEdited(existing)) || markingEdited;
        const patch: Partial<CourseSeedReviewRecord> & {
          reviewStatus: SeedReviewStatus;
        } = {
          ...(body.question !== undefined
            ? { question: body.question, instruction: body.question }
            : {}),
          ...(body.answer !== undefined
            ? { answer: body.answer, response: body.answer }
            : {}),
          ...(body.reviewNotes !== undefined
            ? { reviewNotes: body.reviewNotes }
            : {}),
          reviewStatus,
        };
        if (wasEdited) {
          patch.wasEdited = true;
        }
        if (markingEdited && existing) {
          if (!String(existing.originalQuestion || '').trim()) {
            patch.originalQuestion = exampleQuestion(existing);
          }
          if (!String(existing.originalAnswer || '').trim()) {
            patch.originalAnswer = exampleAnswer(existing);
          }
        }
        return patchSeedLocally(current, seedId, patch);
      });

      try {
        const response = await reviewCourseSeed(courseId, seedId, body);
        setExamples((current) =>
          mergeReviewedSeed(current, seedId, response.seed, reviewStatus),
        );
        setActionMessage(successMessage);
        return true;
      } catch (err) {
        setExamples(snapshot);
        setActionFailed(true);
        setActionMessage(
          err instanceof ApiError
            ? err.message
            : 'That change could not be saved. Try again in a moment.',
        );
        return false;
      } finally {
        setBusyId(null);
      }
    },
    [courseId],
  );

  const approve = useCallback(
    (seedId: string) =>
      applyReviewAction(
        seedId,
        'approved',
        { reviewStatus: 'approved' },
        'Example approved.',
      ),
    [applyReviewAction],
  );

  const reject = useCallback(
    (seedId: string) =>
      applyReviewAction(
        seedId,
        'rejected',
        { reviewStatus: 'rejected' },
        'Example rejected.',
      ),
    [applyReviewAction],
  );

  const saveEdit = useCallback(
    (seedId: string, draft: ReviewDraft) =>
      applyReviewAction(
        seedId,
        'edited',
        {
          reviewStatus: 'edited',
          question: draft.question,
          answer: draft.answer,
          reviewNotes: draft.reviewNotes,
        },
        'Example updated.',
      ),
    [applyReviewAction],
  );

  /*
   * Bulk approve / reject.
   *
   * Deliberately not optimistic. A single review can afford to move the card
   * out of the queue immediately and roll back if the write fails, because the
   * professor is looking straight at it. Fifty cannot: showing forty-eight
   * successes and two silent reversions is exactly the failure mode that makes
   * a professor trust a count that isn't real. So each example is updated only
   * once its own write has come back, and the summary states both numbers.
   */
  const reviewMany = useCallback(
    async (
      seedIds: string[],
      reviewStatus: Extract<SeedReviewStatus, 'approved' | 'rejected'>,
    ): Promise<BulkReviewResult> => {
      const ids = seedIds.map((id) => String(id || '').trim()).filter(Boolean);
      const result: BulkReviewResult = { succeeded: [], failed: [] };
      if (ids.length === 0) {
        return result;
      }

      setBulkBusy(true);
      setActionMessage(null);
      setActionFailed(false);

      try {
        for (let start = 0; start < ids.length; start += BULK_REVIEW_CONCURRENCY) {
          const batch = ids.slice(start, start + BULK_REVIEW_CONCURRENCY);
          const settled = await Promise.all(
            batch.map(async (seedId) => {
              try {
                const response = await reviewCourseSeed(courseId, seedId, {
                  reviewStatus,
                });
                return { seedId, seed: response.seed };
              } catch {
                return { seedId, seed: null };
              }
            }),
          );

          for (const outcome of settled) {
            if (outcome.seed) {
              result.succeeded.push(outcome.seedId);
              setExamples((current) =>
                mergeReviewedSeed(
                  current,
                  outcome.seedId,
                  outcome.seed as CourseSeedReviewRecord,
                  reviewStatus,
                ),
              );
            } else {
              result.failed.push(outcome.seedId);
            }
          }
        }
      } finally {
        setBulkBusy(false);
      }

      const verb = reviewStatus === 'approved' ? 'approved' : 'rejected';
      const done = result.succeeded.length;
      const failed = result.failed.length;
      setActionFailed(failed > 0);
      setActionMessage(
        failed === 0
          ? `${done} ${done === 1 ? 'example' : 'examples'} ${verb}.`
          : `${done} ${verb}. ${failed} could not be saved and ${
              failed === 1 ? 'is' : 'are'
            } unchanged.`,
      );

      return result;
    },
    [courseId],
  );

  return {
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
  };
}
