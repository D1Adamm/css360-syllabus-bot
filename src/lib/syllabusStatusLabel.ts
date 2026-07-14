import type { SyllabusStatus } from '../types';

/** Map stored syllabusStatus values to short course-picker labels. */
export function formatSyllabusStatusLabel(status: SyllabusStatus): string {
  switch (status) {
    case 'indexed':
    case 'ready':
      return 'Indexed';
    case 'extracted':
      return 'Extracted';
    case 'uploaded':
    case 'processing':
      return 'Uploaded';
    case 'upload_failed':
    case 'index_failed':
    case 'error':
      return 'Failed';
    case 'none':
    case 'not_uploaded':
    default:
      return 'Not uploaded';
  }
}
