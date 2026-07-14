import { useId, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FormFieldError } from '../components/FormFieldError';
import { PageHeader } from '../components/PageHeader';
import { createCourse } from '../lib/createCourse';
import { coursePagePath } from '../lib/courseRoutes';

interface FormValues {
  name: string;
  title: string;
  term: string;
  instructorName: string;
}

interface FormErrors {
  name?: string;
  title?: string;
  term?: string;
}

const INITIAL_VALUES: FormValues = {
  name: '',
  title: '',
  term: '',
  instructorName: '',
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

  return errors;
}

export function CreateCoursePage() {
  const formId = useId();
  const navigate = useNavigate();
  const [values, setValues] = useState<FormValues>(INITIAL_VALUES);
  const [errors, setErrors] = useState<FormErrors>({});
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const nameErrorId = `${formId}-name-error`;
  const titleErrorId = `${formId}-title-error`;
  const termErrorId = `${formId}-term-error`;

  function updateField<K extends keyof FormValues>(field: K, value: FormValues[K]) {
    setValues((current) => ({ ...current, [field]: value }));
    setErrors((current) => ({ ...current, [field]: undefined }));
    setSaveError(null);
    setSuccessMessage(null);
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const nextErrors = validate(values);
    setErrors(nextErrors);
    setSaveError(null);
    setSuccessMessage(null);

    if (Object.keys(nextErrors).length > 0) {
      return;
    }

    setSaving(true);

    try {
      const { courseId } = await createCourse({
        name: values.name,
        title: values.title,
        term: values.term,
        instructorName: values.instructorName,
      });

      setSuccessMessage(`Course created successfully. Opening ${courseId}…`);
      navigate(coursePagePath(courseId, 'home'));
    } catch (caughtError) {
      const message =
        caughtError instanceof Error
          ? caughtError.message
          : 'Could not save the course to Firebase.';
      setSaveError(message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <>
      <PageHeader
        title="Create Course"
        description="Create a new course record with a unique course URL. Syllabus upload and processing are not part of this step."
      />

      <aside className="seed-builder-notice" aria-label="Course creation notice">
        <p>
          <strong>Course metadata is stored in Firebase Realtime Database.</strong>
        </p>
        <p>
          After creation you will be taken to <code>/course/{'{courseId}'}/home</code>. Seed
          examples and evaluations for the new course start empty.
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

        {saving && (
          <p className="seed-builder-status" role="status" aria-live="polite">
            Saving course…
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

        <button type="submit" className="seed-form__submit" disabled={saving}>
          {saving ? 'Saving course…' : 'Create course'}
        </button>
      </form>
    </>
  );
}
