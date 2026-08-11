import { useId, useRef, useState } from 'react';
import { Button } from '../ui/Button';
import { FormField } from '../ui/FormField';
import type { SeedExample } from '../../types';
import {
  generateUserSeedId,
  isDuplicateQuestion,
  isMeaningfulText,
  SEED_CATEGORIES,
} from '../../utils/seedDataUtils';

export interface ContributeFormProps {
  /** Existing contributions, used only to catch duplicate questions. */
  existing: SeedExample[];
  /** Section names offered as evidence choices, drawn from the syllabus. */
  sections: string[];
  isSaving: boolean;
  onSubmit: (example: SeedExample) => Promise<void>;
}

interface Values {
  question: string;
  answer: string;
  section: string;
  category: string;
}

const INITIAL: Values = { question: '', answer: '', section: '', category: '' };

type Errors = Partial<Record<keyof Values, string>>;

/**
 * The student contribution form.
 *
 * Reduced to what a student can actually answer: the question, the answer, and
 * optionally where in the syllabus it comes from. Difficulty ratings and the
 * validation vocabulary that used to dominate this form were research
 * metadata, not something a student has an opinion about — the stored record
 * still carries those fields, filled with sensible defaults, so nothing
 * downstream changes.
 *
 * No name or identifier is collected. Contributions are anonymous.
 */
export function ContributeForm({
  existing,
  sections,
  isSaving,
  onSubmit,
}: ContributeFormProps) {
  const formId = useId();
  const [values, setValues] = useState<Values>(INITIAL);
  const [errors, setErrors] = useState<Errors>({});
  const questionRef = useRef<HTMLTextAreaElement>(null);

  function update<K extends keyof Values>(field: K, value: Values[K]) {
    setValues((current) => ({ ...current, [field]: value }));
    setErrors((current) => ({ ...current, [field]: undefined }));
  }

  function validate(): Errors {
    const next: Errors = {};
    const question = values.question.trim();
    const answer = values.answer.trim();

    if (!question) {
      next.question = 'Enter a question.';
    } else if (question.length < 10) {
      next.question = 'Add a little more detail — at least 10 characters.';
    } else if (question.length > 300) {
      next.question = 'Keep the question under 300 characters.';
    } else if (!isMeaningfulText(question)) {
      next.question = 'Enter a real question.';
    } else if (isDuplicateQuestion(question, existing)) {
      next.question = 'Someone has already added this question.';
    }

    if (!answer) {
      next.answer = 'Enter the answer you would expect.';
    } else if (answer.length < 20) {
      next.answer = 'Add a little more detail — at least 20 characters.';
    } else if (answer.length > 1200) {
      next.answer = 'Keep the answer under 1200 characters.';
    } else if (!isMeaningfulText(answer)) {
      next.answer = 'Enter a real answer.';
    }

    return next;
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();

    const nextErrors = validate();
    setErrors(nextErrors);
    if (Object.keys(nextErrors).length > 0) {
      return;
    }

    // The stored shape is unchanged; the fields a student no longer sets get
    // neutral defaults so existing readers and exports keep working.
    const example: SeedExample = {
      id: generateUserSeedId(),
      instruction: values.question.trim(),
      response: values.answer.trim(),
      category: values.category || 'General',
      sourceSection: values.section.trim() || 'Not specified',
      difficulty: 'Medium',
      directlyAnswered: true,
      origin: 'user',
      createdAt: new Date().toISOString(),
    };

    await onSubmit(example);
    setValues(INITIAL);
    setErrors({});
    questionRef.current?.focus();
  }

  return (
    <form className="contribute__form" onSubmit={handleSubmit} noValidate>
      <FormField
        label="Your question"
        hint="Think of something another student might reasonably ask about this course."
        error={errors.question}
        required
      >
        {({ id, describedBy, invalid }) => (
          <textarea
            id={id}
            ref={questionRef}
            aria-describedby={describedBy}
            aria-invalid={invalid}
            value={values.question}
            onChange={(event) => update('question', event.target.value)}
            rows={2}
            maxLength={300}
            disabled={isSaving}
            placeholder="How much of my grade comes from the final project?"
          />
        )}
      </FormField>

      <FormField
        label="The answer you would expect"
        hint="Answer it the way the syllabus does, in your own words."
        error={errors.answer}
        required
      >
        {({ id, describedBy, invalid }) => (
          <textarea
            id={id}
            aria-describedby={describedBy}
            aria-invalid={invalid}
            value={values.answer}
            onChange={(event) => update('answer', event.target.value)}
            rows={5}
            maxLength={1200}
            disabled={isSaving}
            placeholder="The final project is worth 30% of the course grade…"
          />
        )}
      </FormField>

      <div className="contribute__row">
        <FormField
          label="Where in the syllabus is this?"
          hint="Helps your instructor check the answer."
          optional
        >
          {({ id, describedBy }) => (
            <input
              id={id}
              aria-describedby={describedBy}
              list={`${formId}-sections`}
              value={values.section}
              onChange={(event) => update('section', event.target.value)}
              disabled={isSaving}
              placeholder="Grading & Assessment"
              maxLength={120}
            />
          )}
        </FormField>

        <FormField label="Topic" optional>
          {({ id, describedBy }) => (
            <select
              id={id}
              aria-describedby={describedBy}
              value={values.category}
              onChange={(event) => update('category', event.target.value)}
              disabled={isSaving}
            >
              <option value="">Choose a topic…</option>
              {SEED_CATEGORIES.map((category) => (
                <option key={category} value={category}>
                  {category}
                </option>
              ))}
            </select>
          )}
        </FormField>
      </div>

      {/* Section names already used in this course, offered as suggestions. */}
      <datalist id={`${formId}-sections`}>
        {sections.map((section) => (
          <option key={section} value={section} />
        ))}
      </datalist>

      <div className="contribute__submit">
        <Button
          type="submit"
          variant="primary"
          loading={isSaving}
          loadingLabel="Adding…"
          iconLeft="contribute"
        >
          Add question
        </Button>
      </div>
    </form>
  );
}
