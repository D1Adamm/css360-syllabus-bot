import type { CourseSeedReviewRecord } from '../../lib/api';
import type { StatusTone } from '../ui/StatusPill';

/**
 * Wording shared by the two review surfaces.
 *
 * Card view and list view are the same decision made at two different speeds,
 * so a badge must not say "Awaiting review" in one and "generated" in the
 * other. Both read these.
 */

export const REVIEW_STATUS_LABEL: Record<string, string> = {
  generated: 'Awaiting review',
  approved: 'Approved',
  rejected: 'Rejected',
  edited: 'Edited',
};

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
      return 'info';
    default:
      return 'warning';
  }
}

export function exampleSourceLabel(example: CourseSeedReviewRecord): string {
  return String(example.origin || '').trim() === 'user'
    ? 'Student submitted'
    : 'AI generated';
}
