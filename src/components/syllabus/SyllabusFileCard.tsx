import { Icon } from '../ui/Icon';
import { courseSyllabusFileUrl } from '../../lib/api';
import type { CourseMetadata } from '../../types';

export interface SyllabusFileCardProps {
  courseId: string;
  metadata: CourseMetadata | null;
  /** Whether the original file is stored; null while unknown. */
  fileAvailable: boolean | null;
}

const READY = new Set(['indexed', 'ready']);
const FAILED = new Set(['index_failed', 'upload_failed', 'error']);

function statusText(status: string | undefined, chunkCount: number): string {
  if (status && READY.has(status)) {
    return chunkCount > 0
      ? `Ready · ${chunkCount} section${chunkCount === 1 ? '' : 's'} indexed`
      : 'Ready';
  }
  if (status && FAILED.has(status)) {
    return 'The last upload could not be processed';
  }
  if (status === 'processing' || status === 'uploaded' || status === 'extracted') {
    return 'Being prepared';
  }
  return 'No syllabus uploaded yet';
}

/**
 * The current syllabus at a glance: which file, whether it is ready, and the
 * two ways to look at the original. The file is served exactly as uploaded —
 * never reconstructed from the extracted text.
 */
export function SyllabusFileCard({ courseId, metadata, fileAvailable }: SyllabusFileCardProps) {
  const type = metadata?.syllabusType ? metadata.syllabusType.toUpperCase() : null;
  const fileName = metadata?.syllabusFileName || (type ? `syllabus.${type.toLowerCase()}` : null);
  const status = statusText(metadata?.syllabusStatus, metadata?.chunkCount ?? 0);
  const viewUrl = courseSyllabusFileUrl(courseId);
  const downloadUrl = courseSyllabusFileUrl(courseId, true);

  return (
    <div className="syllabus-card">
      <div className="syllabus-card__file">
        <span className="syllabus-card__icon" aria-hidden="true">
          <Icon name="syllabus" size={22} />
        </span>
        <div className="syllabus-card__text">
          <p className="syllabus-card__name">{fileName ?? 'No syllabus file'}</p>
          <p className="syllabus-card__meta">
            {[type, status].filter(Boolean).join(' · ')}
          </p>
        </div>
      </div>

      {fileAvailable && viewUrl && downloadUrl && (
        <div className="syllabus-card__actions">
          <a
            className="ui-button ui-button--secondary ui-button--sm"
            href={viewUrl}
            target="_blank"
            rel="noopener noreferrer"
          >
            <Icon name="reading" size={14} />
            <span className="ui-button__label">View syllabus</span>
          </a>
          <a
            className="ui-button ui-button--tertiary ui-button--sm"
            href={downloadUrl}
            download={fileName ?? undefined}
          >
            <span className="ui-button__label">Download original</span>
          </a>
        </div>
      )}
      {fileAvailable === false && fileName && (
        <p className="ui-text-xs ui-text-muted">
          The original file is not stored for this course. Its extracted text is
          below; replacing the syllabus stores the new file.
        </p>
      )}
    </div>
  );
}
