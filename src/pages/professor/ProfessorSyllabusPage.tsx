import { useEffect, useState } from 'react';
import { ExtractedSyllabusText } from '../../components/syllabus/ExtractedSyllabusText';
import { SyllabusFileCard } from '../../components/syllabus/SyllabusFileCard';
import { SyllabusUploadPanel } from '../../components/upload/SyllabusUploadPanel';
import { Button } from '../../components/ui/Button';
import { PageHeader } from '../../components/ui/PageHeader';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { useCourseId } from '../../context/CourseContext';
import { useCourseMetadata } from '../../hooks/useCourseMetadata';
import { hasCourseSyllabusFile } from '../../lib/api';
import type { SyllabusStatus } from '../../types';

const READY_STATUSES: ReadonlySet<SyllabusStatus> = new Set(['indexed', 'ready']);
const FAILED_STATUSES: ReadonlySet<SyllabusStatus> = new Set([
  'index_failed',
  'upload_failed',
  'error',
]);

/**
 * Syllabus management: which file the course answers from, a way to look at
 * it, a way to replace it, and — on request — the text that was extracted.
 *
 * Compact on purpose. The upload panel stays folded away behind "Replace
 * syllabus" unless the course has no working syllabus, when uploading one is
 * the whole point of the page. Uploading is unchanged: the same panel, the
 * same route, the same guarantee that a failed replacement keeps the current
 * syllabus serving.
 */
export function ProfessorSyllabusPage() {
  const courseId = useCourseId();
  const { metadata, retry } = useCourseMetadata(courseId);
  const [refreshToken, setRefreshToken] = useState(0);
  const [fileAvailable, setFileAvailable] = useState<boolean | null>(null);
  // null: follow the default (open only when there is no working syllabus).
  const [panelChoice, setPanelChoice] = useState<boolean | null>(null);

  const status = metadata?.syllabusStatus;
  const hasSyllabus = status !== undefined && READY_STATUSES.has(status);
  const lastUploadFailed = status !== undefined && FAILED_STATUSES.has(status);
  const panelOpen = panelChoice ?? (metadata !== null && !hasSyllabus);

  useEffect(() => {
    let cancelled = false;
    void hasCourseSyllabusFile(courseId).then((available) => {
      if (!cancelled) {
        setFileAvailable(available);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [courseId, refreshToken]);

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        title="Syllabus"
        description="The syllabus the course assistant answers from."
      />

      <section className="ui-stack" aria-label="Current syllabus">
        <SectionHeader level={2} title="Current syllabus" />
        <SyllabusFileCard courseId={courseId} metadata={metadata} fileAvailable={fileAvailable} />

        {panelOpen ? (
          <SyllabusUploadPanel
            courseId={courseId}
            hasSyllabus={hasSyllabus}
            lastUploadFailed={lastUploadFailed}
            onUploaded={() => {
              setRefreshToken((current) => current + 1);
              retry();
            }}
            onClose={hasSyllabus ? () => setPanelChoice(false) : undefined}
          />
        ) : (
          <div>
            <Button variant="secondary" iconLeft="upload" onClick={() => setPanelChoice(true)}>
              {hasSyllabus ? 'Replace syllabus' : 'Upload syllabus'}
            </Button>
          </div>
        )}

        <ExtractedSyllabusText courseId={courseId} refreshToken={refreshToken} />
      </section>
    </div>
  );
}
