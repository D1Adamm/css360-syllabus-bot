import type { CourseSeedReviewRecord } from '../../lib/api';
import type { StatusTone } from '../ui/StatusPill';

/**
 * Wording shared by the two review surfaces.
 *
 * Card view and list view are the same decision made at two different speeds,
 * so a badge must not say "Awaiting review" in one and "generated" in the
 * other. Both read these.
 */

/**
 * `edited` is reviewed but not decided: the instructor changed the wording and
 * has not yet approved it. Until 2026-11-03 the badge said only "Edited", in
 * the same calm tone as "Approved", while the training export skipped the
 * example and (worse) students were shown it anyway (audit finding B6). The
 * label now says what the status means for the course — the example is not
 * used until approved — and the backend hides it from students on the same
 * rule it uses for the export.
 */
export const EDITED_STATUS_LABEL = 'Edited · not yet approved';

export const REVIEW_STATUS_LABEL: Record<string, string> = {
  generated: 'Awaiting review',
  approved: 'Approved',
  rejected: 'Rejected',
  edited: EDITED_STATUS_LABEL,
};

/** What an edited example's status means, for the page-level notice. */
export function editedAwaitingApprovalNotice(count: number): {
  title: string;
  body: string;
} {
  return {
    title:
      count === 1
        ? '1 edited example is waiting for approval'
        : `${count} edited examples are waiting for approval`,
    body:
      'An edited example is not shown to students and is not included in training data until you approve it. ' +
      'Open the Edited filter and approve each one you are happy with, or use "Save and approve" when editing.',
  };
}

export function reviewStatusLabel(status: string): string {
  return REVIEW_STATUS_LABEL[status] ?? 'Awaiting review';
}

export function reviewStatusTone(status: string): StatusTone {
  switch (status) {
    case 'approved':
      return 'success';
    case 'rejected':
      return 'danger';
    case 'edited':
      // Something still has to happen to it, like "Awaiting review".
      return 'warning';
    default:
      return 'warning';
  }
}

export function exampleSourceLabel(example: CourseSeedReviewRecord): string {
  return String(example.origin || '').trim() === 'user'
    ? 'Student submitted'
    : 'AI generated';
}
