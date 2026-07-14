import { useId, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FormFieldError } from '../components/FormFieldError';
import { PageHeader } from '../components/PageHeader';
import { ApiError, uploadCourseSyllabus } from '../lib/api';
import { updateCourseMetadata } from '../lib/coursesDb';
import { createCourse } from '../lib/createCourse';
import { coursePagePath } from '../lib/courseRoutes';

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

type ProgressState = 'idle' | 'creating' | 'uploading' | 'created';

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
      return 'Creating course…';
    case 'uploading':
      return 'Uploading syllabus…';
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

  const saving = progress === 'creating' || progress === 'uploading';
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

      setProgress('uploading');
      const uploadResult = await uploadCourseSyllabus(courseId, syllabusFile);

      await updateCourseMetadata(courseId, {
        syllabusStatus: 'uploaded',
        syllabusFileName: uploadResult.syllabusFileName,
        syllabusType: uploadResult.syllabusType,
        chunkCount: 0,
      });

      setProgress('created');
      setSuccessMessage(`Course created successfully. Opening ${courseId}…`);
      navigate(coursePagePath(courseId, 'home'));
    } catch (caughtError) {
      if (courseId) {
        try {
          await updateCourseMetadata(courseId, {
            syllabusStatus: 'upload_failed',
            chunkCount: 0,
          });
        } catch {
          // Keep the original upload/create error if metadata rollback fails.
        }
      }

      const message =
        caughtError instanceof ApiError
          ? caughtError.message
          : caughtError instanceof Error
            ? caughtError.message
            : 'Could not create the course or upload the syllabus.';
      setSaveError(message);
      setProgress('idle');
      setSuccessMessage(null);
    }
  }

  return (
    <>
      <PageHeader
        title="Create Course"
        description="Create a new course record, upload a PDF or TXT syllabus, and open its course-specific home page. Text extraction and RAG indexing are not part of this step."
      />

      <aside className="seed-builder-notice" aria-label="Course creation notice">
        <p>
          <strong>Course metadata is stored in Firebase Realtime Database.</strong>
        </p>
        <p>
          The syllabus file is uploaded to the FastAPI backend local course storage. After a
          successful create and upload you will be taken to{' '}
          <code>/course/{'{courseId}'}/home</code>.
        </p>
      </aside>

      <form className="seed-form" onSubmit={handleSubmit} noValidate>
        {successMessage && (
          <p className="seed-form__success" role="status">
            {successMessage}
          </p>
        )}

        {saveError && (
          <p className="seed-builder-status seed-builder-status--error" role="alert">
            {saveError}
          </p>
        )}

        {progressMessage(progress) && (
          <p className="seed-builder-status" role="status" aria-live="polite">
            {progressMessage(progress)}
          </p>
        )}

        <div className="seed-form__field">
          <label htmlFor={`${formId}-name`} className="seed-form__label">
            Course name or code <span className="seed-form__required">(required)</span>
          </label>
          <input
            id={`${formId}-name`}
            className={`seed-form__input${errors.name ? ' seed-form__input--error' : ''}`}
            value={values.name}
            onChange={(event) => updateField('name', event.target.value)}
            placeholder="CSS 430"
            aria-invalid={errors.name ? true : undefined}
            aria-describedby={errors.name ? nameErrorId : undefined}
            disabled={saving}
            maxLength={80}
          />
          {errors.name && <FormFieldError id={nameErrorId} message={errors.name} />}
        </div>

        <div className="seed-form__field">
          <label htmlFor={`${formId}-title`} className="seed-form__label">
            Course title <span className="seed-form__required">(required)</span>
          </label>
          <input
            id={`${formId}-title`}
            className={`seed-form__input${errors.title ? ' seed-form__input--error' : ''}`}
            value={values.title}
            onChange={(event) => updateField('title', event.target.value)}
            placeholder="Operating Systems"
            aria-invalid={errors.title ? true : undefined}
            aria-describedby={errors.title ? titleErrorId : undefined}
            disabled={saving}
            maxLength={160}
          />
          {errors.title && <FormFieldError id={titleErrorId} message={errors.title} />}
        </div>

        <div className="seed-form__field">
          <label htmlFor={`${formId}-term`} className="seed-form__label">
            Term <span className="seed-form__required">(required)</span>
          </label>
          <input
            id={`${formId}-term`}
            className={`seed-form__input${errors.term ? ' seed-form__input--error' : ''}`}
            value={values.term}
            onChange={(event) => updateField('term', event.target.value)}
            placeholder="Summer 2026"
            aria-invalid={errors.term ? true : undefined}
            aria-describedby={errors.term ? termErrorId : undefined}
            disabled={saving}
            maxLength={80}
          />
          {errors.term && <FormFieldError id={termErrorId} message={errors.term} />}
        </div>

        <div className="seed-form__field">
          <label htmlFor={`${formId}-instructor`} className="seed-form__label">
            Instructor name <span className="seed-form__optional">(optional)</span>
          </label>
          <input
            id={`${formId}-instructor`}
            className="seed-form__input"
            value={values.instructorName}
            onChange={(event) => updateField('instructorName', event.target.value)}
            placeholder="Instructor name"
            disabled={saving}
            maxLength={120}
          />
        </div>

        <div className="seed-form__field">
          <label htmlFor={`${formId}-syllabus`} className="seed-form__label">
            Syllabus file <span className="seed-form__required">(required)</span>
          </label>
          <input
            id={`${formId}-syllabus`}
            className={`seed-form__input${errors.syllabusFile ? ' seed-form__input--error' : ''}`}
            type="file"
            accept=".pdf,.txt,application/pdf,text/plain"
            onChange={(event) =>
              updateField('syllabusFile', event.target.files?.[0] ?? null)
            }
            aria-invalid={errors.syllabusFile ? true : undefined}
            aria-describedby={errors.syllabusFile ? syllabusErrorId : undefined}
            disabled={saving}
          />
          {errors.syllabusFile && (
            <FormFieldError id={syllabusErrorId} message={errors.syllabusFile} />
          )}
        </div>

        <button type="submit" className="seed-form__submit" disabled={saving}>
          {saving ? progressMessage(progress) ?? 'Working…' : 'Create course'}
        </button>
      </form>
    </>
  );
}
