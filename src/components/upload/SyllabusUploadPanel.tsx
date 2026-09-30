import { useId, useState } from 'react';
import { Button } from '../ui/Button';
import { Callout } from '../ui/Callout';
import { SectionHeader } from '../ui/SectionHeader';
import { Surface } from '../ui/Surface';
import { uploadCourseSyllabus, type SyllabusUploadResponse } from '../../lib/api';
import { toUserMessage } from '../../lib/errorMessages';
import { validateSyllabusFile } from '../../lib/syllabusFile';
import { SyllabusDropzone } from './SyllabusDropzone';

export interface SyllabusUploadPanelProps {
  courseId: string;
  /** The course has a processed syllabus that students and the assistant use. */
  hasSyllabus: boolean;
  /** The last upload into a course with no syllabus could not be processed. */
  lastUploadFailed: boolean;
  onUploaded: (result: SyllabusUploadResponse) => void;
  /** Fold the panel away again. Omitted when there is nothing to fall back to. */
  onClose?: () => void;
}

/**
 * Upload a course's syllabus, retry a failed one, or replace it.
 *
 * All three are the same request to the same route. The backend writes nothing
 * until the new file has been fully processed, so a failed replacement leaves
 * the current syllabus in use — which is what the error here says.
 */
export function SyllabusUploadPanel({
  courseId,
  hasSyllabus,
  lastUploadFailed,
  onUploaded,
  onClose,
}: SyllabusUploadPanelProps) {
  const errorId = useId();
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | undefined>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const problem = validateSyllabusFile(file);
    setFileError(problem);
    setError(null);
    setDone(null);
    if (problem || !file) {
      return;
    }

    setBusy(true);
    try {
      const result = await uploadCourseSyllabus(courseId, file);
      setFile(null);
      setDone(
        result.replaced
          ? 'Syllabus replaced. Students and the course assistant now use the new version.'
          : 'Syllabus uploaded. Students and the course assistant now use it.',
      );
      onUploaded(result);
    } catch (caught) {
      const message = toUserMessage(caught, {
        audience: 'professor',
        context: 'syllabus-upload',
      }).message;
      setError(hasSyllabus ? `${message} Your current syllabus is still in use.` : message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Surface as="section" tone="sunken" padding="md" aria-label="Syllabus upload">
      <form className="ui-stack" onSubmit={handleSubmit} noValidate>
        <SectionHeader
          level={2}
          title={hasSyllabus ? 'Replace the syllabus' : 'Upload the syllabus'}
          description={
            hasSyllabus
              ? 'Upload a new version. It replaces the current one for students and the course assistant once it has been processed.'
              : 'Students and the course assistant use the syllabus once it has been processed.'
          }
        />

        {lastUploadFailed && !hasSyllabus && !done && (
          <Callout tone="warning" title="The last syllabus upload could not be processed">
            Upload it again to finish setting up this course.
          </Callout>
        )}
        {error && (
          <Callout tone="danger" title="We couldn't process this syllabus">
            {error}
          </Callout>
        )}
        {done && <Callout tone="success">{done}</Callout>}

        <SyllabusDropzone
          file={file}
          onSelect={(next) => {
            setFile(next);
            setFileError(undefined);
            setError(null);
            setDone(null);
          }}
          disabled={busy}
          error={fileError}
          errorId={errorId}
        />

        <div className="ui-row">
          <Button
            type="submit"
            variant="primary"
            loading={busy}
            loadingLabel="Processing the syllabus…"
          >
            {hasSyllabus ? 'Replace syllabus' : 'Upload syllabus'}
          </Button>
          {onClose && (
            <Button type="button" variant="tertiary" onClick={onClose} disabled={busy}>
              {done ? 'Done' : 'Cancel'}
            </Button>
          )}
        </div>
      </form>
    </Surface>
  );
}
