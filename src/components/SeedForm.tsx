import { useId, useRef, useState } from 'react';
import seedData from '../data/seedData.json';
import syllabusTopics from '../data/syllabusTopics.json';
import type { SeedDifficulty, SeedExample } from '../types';
import {
  generateUserSeedId,
  isDuplicateQuestion,
  isMeaningfulText,
  SEED_CATEGORIES,
} from '../utils/seedDataUtils';
import { FormFieldError } from './FormFieldError';

const prototypeSeeds = seedData as SeedExample[];

const DIFFICULTIES: SeedDifficulty[] = ['Easy', 'Medium', 'Hard'];

const SOURCE_SECTION_CUSTOM = '__custom__';

interface SeedFormProps {
  userSeeds: SeedExample[];
  onAddSeed: (seed: SeedExample) => Promise<void>;
  isSaving?: boolean;
  isLoading?: boolean;
}

interface FormValues {
  instruction: string;
  response: string;
  category: string;
  sourceSection: string;
  customSourceSection: string;
  difficulty: SeedDifficulty | '';
  directlyAnswered: string;
  notes: string;
}

interface FormErrors {
  instruction?: string;
  response?: string;
  category?: string;
  sourceSection?: string;
  customSourceSection?: string;
  difficulty?: string;
  directlyAnswered?: string;
}

const INITIAL_VALUES: FormValues = {
  instruction: '',
  response: '',
  category: '',
  sourceSection: '',
  customSourceSection: '',
  difficulty: '',
  directlyAnswered: '',
  notes: '',
};

function getSourceSectionOptions(): string[] {
  const fromSeeds = prototypeSeeds.map((seed) => seed.sourceSection);
  const fromTopics = (syllabusTopics as { sourceSection: string }[]).map(
    (topic) => topic.sourceSection,
  );
  return Array.from(new Set([...fromSeeds, ...fromTopics])).sort((a, b) =>
    a.localeCompare(b),
  );
}

const sourceSectionOptions = getSourceSectionOptions();

export function SeedForm({
  userSeeds,
  onAddSeed,
  isSaving = false,
  isLoading = false,
}: SeedFormProps) {
  const formId = useId();
  const [values, setValues] = useState<FormValues>(INITIAL_VALUES);
  const [errors, setErrors] = useState<FormErrors>({});
  const [successMessage, setSuccessMessage] = useState('');
  const instructionRef = useRef<HTMLTextAreaElement>(null);

  const instructionErrorId = `${formId}-instruction-error`;
  const responseErrorId = `${formId}-response-error`;
  const categoryErrorId = `${formId}-category-error`;
  const sourceSectionErrorId = `${formId}-source-section-error`;
  const customSourceSectionErrorId = `${formId}-custom-source-section-error`;
  const difficultyErrorId = `${formId}-difficulty-error`;
  const directlyAnsweredErrorId = `${formId}-directly-answered-error`;

  function updateField<K extends keyof FormValues>(field: K, value: FormValues[K]) {
    setValues((current) => ({ ...current, [field]: value }));
    setErrors((current) => ({ ...current, [field]: undefined }));
    setSuccessMessage('');
  }

  function validate(): FormErrors {
    const nextErrors: FormErrors = {};
    const instruction = values.instruction.trim();
    const response = values.response.trim();
    const resolvedSourceSection =
      values.sourceSection === SOURCE_SECTION_CUSTOM
        ? values.customSourceSection.trim()
        : values.sourceSection.trim();

    if (!instruction) {
      nextErrors.instruction = 'A syllabus question is required.';
    } else if (instruction.length < 10) {
      nextErrors.instruction = 'The question must be at least 10 characters.';
    } else if (instruction.length > 300) {
      nextErrors.instruction = 'The question must be 300 characters or fewer.';
    } else if (!isMeaningfulText(instruction)) {
      nextErrors.instruction = 'Enter a meaningful question, not only punctuation.';
    } else if (isDuplicateQuestion(instruction, [...prototypeSeeds, ...userSeeds])) {
      nextErrors.instruction = 'This question already exists in the dataset.';
    }

    if (!response) {
      nextErrors.response = 'An expected answer is required.';
    } else if (response.length < 20) {
      nextErrors.response = 'The expected answer must be at least 20 characters.';
    } else if (response.length > 1200) {
      nextErrors.response = 'The expected answer must be 1200 characters or fewer.';
    } else if (!isMeaningfulText(response)) {
      nextErrors.response = 'Enter a meaningful answer, not only punctuation.';
    }

    if (!values.category) {
      nextErrors.category = 'Select a category.';
    }

    if (!values.sourceSection) {
      nextErrors.sourceSection = 'Select a source section.';
    } else if (
      values.sourceSection === SOURCE_SECTION_CUSTOM &&
      !values.customSourceSection.trim()
    ) {
      nextErrors.customSourceSection = 'Enter a custom source section.';
    } else if (
      values.sourceSection === SOURCE_SECTION_CUSTOM &&
      !isMeaningfulText(values.customSourceSection)
    ) {
      nextErrors.customSourceSection =
        'Enter a meaningful source section, not only punctuation.';
    }

    if (!values.difficulty) {
      nextErrors.difficulty = 'Select a difficulty level.';
    }

    if (!values.directlyAnswered) {
      nextErrors.directlyAnswered = 'Indicate whether the syllabus directly answers this.';
    }

    if (!resolvedSourceSection && values.sourceSection !== SOURCE_SECTION_CUSTOM) {
      nextErrors.sourceSection = 'Select a source section.';
    }

    return nextErrors;
  }

  function focusFirstError(nextErrors: FormErrors) {
    if (nextErrors.instruction) {
      instructionRef.current?.focus();
      return;
    }

    const firstErrorField = document.querySelector<HTMLElement>('[aria-invalid="true"]');
    firstErrorField?.focus();
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextErrors = validate();

    if (Object.keys(nextErrors).length > 0) {
      setErrors(nextErrors);
      focusFirstError(nextErrors);
      return;
    }

    const resolvedSourceSection =
      values.sourceSection === SOURCE_SECTION_CUSTOM
        ? values.customSourceSection.trim()
        : values.sourceSection.trim();

    const newSeed: SeedExample = {
      id: generateUserSeedId(),
      instruction: values.instruction.trim(),
      response: values.response.trim(),
      category: values.category,
      sourceSection: resolvedSourceSection,
      difficulty: values.difficulty as SeedDifficulty,
      directlyAnswered: values.directlyAnswered === 'yes',
      origin: 'user',
      createdAt: new Date().toISOString(),
      ...(values.notes.trim() ? { notes: values.notes.trim() } : {}),
    };

    try {
      await onAddSeed(newSeed);
      setValues(INITIAL_VALUES);
      setErrors({});
      setSuccessMessage('Your example was saved to the shared dataset.');
      instructionRef.current?.focus();
    } catch {
      setSuccessMessage('');
    }
  }

  return (
    <form className="seed-form" onSubmit={handleSubmit} noValidate>
      {successMessage && (
        <p className="seed-form__success" role="status">
          {successMessage}
        </p>
      )}

      <div className="seed-form__field">
        <label htmlFor={`${formId}-instruction`} className="seed-form__label">
          Syllabus question <span className="seed-form__required">(required)</span>
        </label>
        <textarea
          ref={instructionRef}
          id={`${formId}-instruction`}
          className={`seed-form__textarea${errors.instruction ? ' seed-form__input--error' : ''}`}
          value={values.instruction}
          onChange={(event) => updateField('instruction', event.target.value)}
          placeholder="Can I make up an in-class activity if I miss class?"
          rows={3}
          aria-invalid={errors.instruction ? true : undefined}
          aria-describedby={errors.instruction ? instructionErrorId : undefined}
          maxLength={300}
        />
        {errors.instruction && (
          <FormFieldError id={instructionErrorId} message={errors.instruction} />
        )}
      </div>

      <div className="seed-form__field">
        <label htmlFor={`${formId}-response`} className="seed-form__label">
          Expected answer <span className="seed-form__required">(required)</span>
        </label>
        <textarea
          id={`${formId}-response`}
          className={`seed-form__textarea${errors.response ? ' seed-form__input--error' : ''}`}
          value={values.response}
          onChange={(event) => updateField('response', event.target.value)}
          rows={5}
          aria-invalid={errors.response ? true : undefined}
          aria-describedby={errors.response ? responseErrorId : undefined}
          maxLength={1200}
        />
        {errors.response && <FormFieldError id={responseErrorId} message={errors.response} />}
      </div>

      <div className="seed-form__row">
        <div className="seed-form__field">
          <label htmlFor={`${formId}-category`} className="seed-form__label">
            Category <span className="seed-form__required">(required)</span>
          </label>
          <select
            id={`${formId}-category`}
            className={`seed-form__select${errors.category ? ' seed-form__input--error' : ''}`}
            value={values.category}
            onChange={(event) => updateField('category', event.target.value)}
            aria-invalid={errors.category ? true : undefined}
            aria-describedby={errors.category ? categoryErrorId : undefined}
          >
            <option value="">Select a category</option>
            {SEED_CATEGORIES.map((category) => (
              <option key={category} value={category}>
                {category}
              </option>
            ))}
          </select>
          {errors.category && <FormFieldError id={categoryErrorId} message={errors.category} />}
        </div>

        <div className="seed-form__field">
          <label htmlFor={`${formId}-difficulty`} className="seed-form__label">
            Difficulty <span className="seed-form__required">(required)</span>
          </label>
          <select
            id={`${formId}-difficulty`}
            className={`seed-form__select${errors.difficulty ? ' seed-form__input--error' : ''}`}
            value={values.difficulty}
            onChange={(event) =>
              updateField('difficulty', event.target.value as SeedDifficulty | '')
            }
            aria-invalid={errors.difficulty ? true : undefined}
            aria-describedby={errors.difficulty ? difficultyErrorId : undefined}
          >
            <option value="">Select difficulty</option>
            {DIFFICULTIES.map((difficulty) => (
              <option key={difficulty} value={difficulty}>
                {difficulty}
              </option>
            ))}
          </select>
          {errors.difficulty && (
            <FormFieldError id={difficultyErrorId} message={errors.difficulty} />
          )}
        </div>
      </div>

      <div className="seed-form__field">
        <label htmlFor={`${formId}-source-section`} className="seed-form__label">
          Source section <span className="seed-form__required">(required)</span>
        </label>
        <select
          id={`${formId}-source-section`}
          className={`seed-form__select${errors.sourceSection ? ' seed-form__input--error' : ''}`}
          value={values.sourceSection}
          onChange={(event) => updateField('sourceSection', event.target.value)}
          aria-invalid={errors.sourceSection ? true : undefined}
          aria-describedby={errors.sourceSection ? sourceSectionErrorId : undefined}
        >
          <option value="">Select a syllabus section</option>
          {sourceSectionOptions.map((section) => (
            <option key={section} value={section}>
              {section}
            </option>
          ))}
          <option value={SOURCE_SECTION_CUSTOM}>Custom section…</option>
        </select>
        {errors.sourceSection && (
          <FormFieldError id={sourceSectionErrorId} message={errors.sourceSection} />
        )}
      </div>

      {values.sourceSection === SOURCE_SECTION_CUSTOM && (
        <div className="seed-form__field">
          <label htmlFor={`${formId}-custom-source-section`} className="seed-form__label">
            Custom source section <span className="seed-form__required">(required)</span>
          </label>
          <input
            id={`${formId}-custom-source-section`}
            type="text"
            className={`seed-form__input${errors.customSourceSection ? ' seed-form__input--error' : ''}`}
            value={values.customSourceSection}
            onChange={(event) => updateField('customSourceSection', event.target.value)}
            aria-invalid={errors.customSourceSection ? true : undefined}
            aria-describedby={
              errors.customSourceSection ? customSourceSectionErrorId : undefined
            }
          />
          {errors.customSourceSection && (
            <FormFieldError
              id={customSourceSectionErrorId}
              message={errors.customSourceSection}
            />
          )}
        </div>
      )}

      <fieldset
        className="seed-form__fieldset"
        aria-describedby={errors.directlyAnswered ? directlyAnsweredErrorId : undefined}
      >
        <legend className="seed-form__label">
          Directly answered <span className="seed-form__required">(required)</span>
        </legend>
        <div className="seed-form__radio-group">
          <label className="seed-form__radio-label">
            <input
              type="radio"
              name={`${formId}-directly-answered`}
              value="yes"
              checked={values.directlyAnswered === 'yes'}
              onChange={(event) => updateField('directlyAnswered', event.target.value)}
            />
            Yes, the syllabus directly answers this
          </label>
          <label className="seed-form__radio-label">
            <input
              type="radio"
              name={`${formId}-directly-answered`}
              value="no"
              checked={values.directlyAnswered === 'no'}
              onChange={(event) => updateField('directlyAnswered', event.target.value)}
            />
            No, clarification or another source is needed
          </label>
        </div>
        {errors.directlyAnswered && (
          <FormFieldError id={directlyAnsweredErrorId} message={errors.directlyAnswered} />
        )}
      </fieldset>

      <div className="seed-form__field">
        <label htmlFor={`${formId}-notes`} className="seed-form__label">
          Notes or reasoning <span className="seed-form__optional">(optional)</span>
        </label>
        <textarea
          id={`${formId}-notes`}
          className="seed-form__textarea"
          value={values.notes}
          onChange={(event) => updateField('notes', event.target.value)}
          rows={3}
          placeholder="Explain why this answer is appropriate or what distinction this example teaches."
        />
      </div>

      <button
        type="submit"
        className="seed-form__submit"
        disabled={isSaving || isLoading}
      >
        {isSaving ? 'Saving example…' : 'Save example'}
      </button>
    </form>
  );
}
