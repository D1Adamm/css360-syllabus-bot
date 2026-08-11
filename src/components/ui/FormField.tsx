import { useId } from 'react';

export interface FormFieldRenderProps {
  /** Apply to the control. */
  id: string;
  /** Apply as `aria-describedby`; already combines hint and error ids. */
  describedBy: string | undefined;
  /** Apply as `aria-invalid`. */
  invalid: boolean | undefined;
}

export interface FormFieldProps {
  label: string;
  /** Guidance shown before the user makes a mistake, not after. */
  hint?: string;
  error?: string;
  required?: boolean;
  /** Marks a field explicitly optional, which is clearer than silence. */
  optional?: boolean;
  className?: string;
  children: React.ReactNode | ((props: FormFieldRenderProps) => React.ReactNode);
}

/**
 * Label, hint, control and error as one accessible unit.
 *
 * The three hand-rolled form systems in the legacy pages each re-derived
 * `useId`-suffixed error ids and wired `aria-describedby` by hand. This does it
 * once. Pass a function child to receive the generated ids:
 *
 *   <FormField label="Question" error={errors.question}>
 *     {({ id, describedBy, invalid }) => (
 *       <textarea id={id} aria-describedby={describedBy} aria-invalid={invalid} />
 *     )}
 *   </FormField>
 */
export function FormField({
  label,
  hint,
  error,
  required = false,
  optional = false,
  className,
  children,
}: FormFieldProps) {
  const baseId = useId();
  const id = `${baseId}-control`;
  const hintId = `${baseId}-hint`;
  const errorId = `${baseId}-error`;

  const describedBy =
    [hint ? hintId : null, error ? errorId : null].filter(Boolean).join(' ') || undefined;

  const classes = ['ui-field', error ? 'ui-field--invalid' : '', className]
    .filter(Boolean)
    .join(' ');

  return (
    <div className={classes}>
      <label className="ui-field__label" htmlFor={id}>
        {label}
        {required && (
          <span className="ui-field__requirement"> (required)</span>
        )}
        {optional && !required && (
          <span className="ui-field__requirement"> (optional)</span>
        )}
      </label>

      {hint && (
        <p className="ui-field__hint" id={hintId}>
          {hint}
        </p>
      )}

      <div className="ui-field__control">
        {typeof children === 'function'
          ? children({ id, describedBy, invalid: error ? true : undefined })
          : children}
      </div>

      {error && (
        <p className="ui-field__error" id={errorId} role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
