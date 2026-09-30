import { useState } from 'react';
import { SyllabusView } from '../../components/syllabus/SyllabusView';
import { SyllabusUploadPanel } from '../../components/upload/SyllabusUploadPanel';
import { useCourseId } from '../../context/CourseContext';
import { useCourseMetadata } from '../../hooks/useCourseMetadata';
import type { SyllabusStatus } from '../../types';

const READY_STATUSES: ReadonlySet<SyllabusStatus> = new Set(['indexed', 'ready']);
const FAILED_STATUSES: ReadonlySet<SyllabusStatus> = new Set([
  'index_failed',
  'upload_failed',
  'error',
]);

export function ProfessorSyllabusPage() {
  const courseId = useCourseId();
  const { metadata, retry } = useCourseMetadata(courseId);
  const [refreshToken, setRefreshToken] = useState(0);
  const status = metadata?.syllabusStatus;

  return (
    <SyllabusView audience="professor" refreshToken={refreshToken}>
      <SyllabusUploadPanel
        courseId={courseId}
        hasSyllabus={status !== undefined && READY_STATUSES.has(status)}
        lastUploadFailed={status !== undefined && FAILED_STATUSES.has(status)}
        onUploaded={() => {
          setRefreshToken((current) => current + 1);
          retry();
        }}
      />
    </SyllabusView>
  );
}
