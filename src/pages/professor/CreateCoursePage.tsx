import { useId, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FormFieldError } from '../../components/FormFieldError';
import { SyllabusDropzone } from '../../components/upload/SyllabusDropzone';
import { Button, LinkButton } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { PageHeader } from '../../components/ui/PageHeader';
import { useSession } from '../../context/SessionContext';
import { uploadCourseSyllabus } from '../../lib/api';
import { createCourse } from '../../lib/createCourse';
import { toUserMessage } from '../../lib/errorMessages';
import { professorCourseHomePath, professorCoursePath } from '../../lib/roleRoutes';
import { validateSyllabusFile } from '../../lib/syllabusFile';

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
  const syllabusError = validateSyllabusFile(values.syllabusFile);
  if (syllabusError) {
    errors.syllabusFile = syllabusError;
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
  const { refresh: refreshSession } = useSession();
  const [values, setValues] = useState<FormValues>(INITIAL_VALUES);
  const [errors, setErrors] = useState<FormErrors>({});
  const [progress, setProgress] = useState<ProgressState>('idle');
  const [saveError, setSaveError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  // Set once the course exists. From then on a submit only retries the
  // syllabus upload into it: a failed upload must never lead to a duplicate.
  const [createdCourseId, setCreatedCourseId] = useState<string | null>(null);

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
    let courseId = createdCourseId;

    try {
      if (!courseId) {
        setProgress('creating');
        const created = await createCourse({
          name: values.name,
          title: values.title,
          term: values.term,
          instructorName: values.instructorName,
        });
        courseId = created.courseId;
        setCreatedCourseId(courseId);
        // The backend made this professor the course's instructor in the same
        // transaction. The cached session predates that membership, and the
        // course routes' guard reads it, so ask again before navigating there.
        await refreshSession();
      }

      // The upload records the syllabus state on the course itself, success
      // or failure, so there is nothing left for this page to write.
      setProgress('indexing');
      await uploadCourseSyllabus(courseId, syllabusFile);

      setProgress('created');
      setSuccessMessage('Course created. Opening it now…');
      navigate(professorCourseHomePath(courseId));
    } catch (caughtError) {
      // Upload failures often carry a usable validation message; anything else
      // becomes role-appropriate copy rather than raw infrastructure text.
      setSaveError(
        toUserMessage(caughtError, {
          audience: 'professor',
          context: courseId ? 'syllabus-upload' : 'course-create',
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
        {saveError && !createdCourseId && (
          <Callout tone="danger" title="We couldn't create this course">
            {saveError}
          </Callout>
        )}

        {saveError && createdCourseId && (
          <Callout
            tone="danger"
            title="Your course was created, but its syllabus could not be processed"
            actions={
              <LinkButton to={professorCoursePath(createdCourseId, 'syllabus')} variant="secondary">
                Open the course
              </LinkButton>
            }
          >
            {saveError} Upload it again below and it goes into the same course, or open
            the course and add the syllabus from its Syllabus page later.
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
                disabled={saving || createdCourseId !== null}
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
                disabled={saving || createdCourseId !== null}
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
              disabled={saving || createdCourseId !== null}
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
              disabled={saving || createdCourseId !== null}
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
            {createdCourseId ? 'Upload syllabus' : 'Create course'}
          </Button>
        </div>
      </form>
    </div>
  );
}
