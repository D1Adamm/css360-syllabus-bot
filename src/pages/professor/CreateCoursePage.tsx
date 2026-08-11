import { useId, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FormFieldError } from '../../components/FormFieldError';
import { SyllabusDropzone } from '../../components/upload/SyllabusDropzone';
import { Button } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { PageHeader } from '../../components/ui/PageHeader';
import { uploadCourseSyllabus } from '../../lib/api';
import { updateCourseMetadata } from '../../lib/coursesDb';
import { createCourse } from '../../lib/createCourse';
import { toUserMessage } from '../../lib/errorMessages';
import { professorCourseHomePath } from '../../lib/roleRoutes';

interface FormValues {
  name: string;
  title: string;
  term: string;
  instructorName: string;
  syllabusFile: File | null;
}

interface FormErrors {
  name?: string;
  title?: string;
  term?: string;
  syllabusFile?: string;
}

type ProgressState = 'idle' | 'creating' | 'uploading' | 'indexing' | 'created';

const INITIAL_VALUES: FormValues = {
  name: '',
  title: '',
  term: '',
  instructorName: '',
  syllabusFile: null,
};

const ALLOWED_SYLLABUS_EXTENSIONS = new Set(['pdf', 'txt']);

function getFileExtension(fileName: string): string {
  const parts = fileName.toLowerCase().split('.');
  return parts.length > 1 ? (parts.at(-1) ?? '') : '';
}

function validate(values: FormValues): FormErrors {
  const errors: FormErrors = {};

  if (!values.name.trim()) {
    errors.name = 'Course name or code is required.';
  }
  if (!values.title.trim()) {
    errors.title = 'Course title is required.';
  }
  if (!values.term.trim()) {
    errors.term = 'Term is required.';
  }
  if (!values.syllabusFile) {
    errors.syllabusFile = 'A PDF or TXT syllabus file is required.';
  } else {
    const extension = getFileExtension(values.syllabusFile.name);
    if (!ALLOWED_SYLLABUS_EXTENSIONS.has(extension)) {
      errors.syllabusFile = 'Only .pdf and .txt syllabus files are supported.';
    }
  }

  return errors;
}

function progressMessage(progress: ProgressState): string | null {
  switch (progress) {
    case 'creating':
      return 'Creating your course…';
    case 'uploading':
      return 'Uploading your syllabus…';
    case 'indexing':
      return 'Preparing your syllabus…';
    case 'created':
      return 'Course created';
    default:
      return null;
  }
}

export function CreateCoursePage() {
  const formId = useId();
  const navigate = useNavigate();
  const [values, setValues] = useState<FormValues>(INITIAL_VALUES);
  const [errors, setErrors] = useState<FormErrors>({});
  const [progress, setProgress] = useState<ProgressState>('idle');
  const [saveError, setSaveError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const saving =
    progress === 'creating' || progress === 'uploading' || progress === 'indexing';
  const nameErrorId = `${formId}-name-error`;
  const titleErrorId = `${formId}-title-error`;
  const termErrorId = `${formId}-term-error`;
  const syllabusErrorId = `${formId}-syllabus-error`;

  function updateField<K extends keyof FormValues>(field: K, value: FormValues[K]) {
    setValues((current) => ({ ...current, [field]: value }));
    setErrors((current) => ({ ...current, [field]: undefined }));
    setSaveError(null);
    setSuccessMessage(null);
    if (progress === 'created') {
      setProgress('idle');
    }
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const nextErrors = validate(values);
    setErrors(nextErrors);
    setSaveError(null);
    setSuccessMessage(null);

    if (Object.keys(nextErrors).length > 0 || !values.syllabusFile) {
      return;
    }

    const syllabusFile = values.syllabusFile;
    let courseId: string | null = null;

    try {
      setProgress('creating');
      const created = await createCourse({
        name: values.name,
        title: values.title,
        term: values.term,
        instructorName: values.instructorName,
      });
      courseId = created.courseId;

      setProgress('indexing');
      const uploadResult = await uploadCourseSyllabus(courseId, syllabusFile);

      await updateCourseMetadata(courseId, {
        syllabusStatus: 'indexed',
        syllabusFileName: uploadResult.syllabusFileName,
        syllabusType: uploadResult.syllabusType,
        chunkCount: uploadResult.chunkCount,
      });

      setProgress('created');
      setSuccessMessage('Course created. Opening it now…');
      navigate(professorCourseHomePath(courseId));
    } catch (caughtError) {
      if (courseId) {
        try {
          await updateCourseMetadata(courseId, {
            syllabusStatus: 'index_failed',
            chunkCount: 0,
          });
        } catch {
          // Keep the original upload/create error if metadata rollback fails.
        }
      }

      // Upload failures often carry a usable validation message; anything else
      // becomes role-appropriate copy rather than raw infrastructure text.
      setSaveError(
        toUserMessage(caughtError, {
          audience: 'professor',
          context: 'syllabus-upload',
        }).message,
      );
      setProgress('idle');
      setSuccessMessage(null);
    }
  }

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        title="Create a course"
        description="Add your course and upload its syllabus. We'll prepare it so the assistant can answer questions from it."
      />

      <form className="course-form" onSubmit={handleSubmit} noValidate>
        {saveError && (
          <Callout tone="danger" title="We couldn't create this course">
            {saveError}
          </Callout>
        )}

        {successMessage && <Callout tone="success">{successMessage}</Callout>}

        {saving && progressMessage(progress) && (
          <Callout tone="info" live>
            {progressMessage(progress)}
          </Callout>
        )}

        <div className="course-form__grid">
          <div className="ui-field">
            <label htmlFor={`${formId}-name`} className="ui-field__label">
              Course name or code{' '}
              <span className="ui-field__requirement">(required)</span>
            </label>
            <div className="ui-field__control">
              <input
                id={`${formId}-name`}
                value={values.name}
                onChange={(event) => updateField('name', event.target.value)}
                placeholder="CSS 430"
                aria-invalid={errors.name ? true : undefined}
                aria-describedby={errors.name ? nameErrorId : undefined}
                disabled={saving}
                maxLength={80}
              />
            </div>
            {errors.name && <FormFieldError id={nameErrorId} message={errors.name} />}
          </div>

          <div className="ui-field">
            <label htmlFor={`${formId}-term`} className="ui-field__label">
              Term <span className="ui-field__requirement">(required)</span>
            </label>
            <div className="ui-field__control">
              <input
                id={`${formId}-term`}
                value={values.term}
                onChange={(event) => updateField('term', event.target.value)}
                placeholder="Summer 2026"
                aria-invalid={errors.term ? true : undefined}
                aria-describedby={errors.term ? termErrorId : undefined}
                disabled={saving}
                maxLength={80}
              />
            </div>
            {errors.term && <FormFieldError id={termErrorId} message={errors.term} />}
          </div>
        </div>

        <div className="ui-field">
          <label htmlFor={`${formId}-title`} className="ui-field__label">
            Course title <span className="ui-field__requirement">(required)</span>
          </label>
          <div className="ui-field__control">
            <input
              id={`${formId}-title`}
              value={values.title}
              onChange={(event) => updateField('title', event.target.value)}
              placeholder="Operating Systems"
              aria-invalid={errors.title ? true : undefined}
              aria-describedby={errors.title ? titleErrorId : undefined}
              disabled={saving}
              maxLength={160}
            />
          </div>
          {errors.title && <FormFieldError id={titleErrorId} message={errors.title} />}
        </div>

        <div className="ui-field">
          <label htmlFor={`${formId}-instructor`} className="ui-field__label">
            Instructor name <span className="ui-field__requirement">(optional)</span>
          </label>
          <div className="ui-field__control">
            <input
              id={`${formId}-instructor`}
              value={values.instructorName}
              onChange={(event) => updateField('instructorName', event.target.value)}
              placeholder="Shown to students on the course page"
              disabled={saving}
              maxLength={120}
            />
          </div>
        </div>

        <SyllabusDropzone
          file={values.syllabusFile}
          onSelect={(file) => updateField('syllabusFile', file)}
          disabled={saving}
          error={errors.syllabusFile}
          errorId={syllabusErrorId}
        />

        <div className="course-form__submit">
          <Button
            type="submit"
            variant="primary"
            loading={saving}
            loadingLabel={progressMessage(progress) ?? 'Working…'}
          >
            Create course
          </Button>
        </div>
      </form>
    </div>
  );
}
