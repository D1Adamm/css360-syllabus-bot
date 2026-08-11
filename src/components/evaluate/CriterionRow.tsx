import { APPROACHES } from '../compare/approaches';
import { FormFieldError } from '../FormFieldError';
import type { ModelKey } from '../../types';

export interface CriterionRowProps {
  legend: string;
  hint?: string;
  name: string;
  value: ModelKey | '';
  error?: string;
  errorId: string;
  disabled?: boolean;
  /** Approaches that could not answer, so they cannot be chosen. */
  unavailable: ModelKey[];
  onChange: (value: ModelKey) => void;
  fieldsetRef?: React.Ref<HTMLFieldSetElement>;
}

/**
 * One rating criterion as a row of four choices, labelled A–D to match the
 * responses above. Choosing by marker keeps the row compact enough that all
 * five criteria fit on one screen alongside the answers.
 */
export function CriterionRow({
  legend,
  hint,
  name,
  value,
  error,
  errorId,
  disabled = false,
  unavailable,
  onChange,
  fieldsetRef,
}: CriterionRowProps) {
  return (
    <fieldset
      className={`criterion${error ? ' criterion--invalid' : ''}`}
      ref={fieldsetRef}
      tabIndex={error ? -1 : undefined}
    >
      <legend className="criterion__legend">{legend}</legend>
      {hint && <p className="criterion__hint">{hint}</p>}

      <div className="criterion__options" role="radiogroup" aria-label={legend}>
        {APPROACHES.map((approach, index) => {
          const marker = String.fromCharCode(65 + index);
          const inputId = `${name}-${approach.key}`;
          const isUnavailable = unavailable.includes(approach.key);

          return (
            <label
              key={approach.key}
              htmlFor={inputId}
              className={`criterion__option${
                value === approach.key ? ' criterion__option--selected' : ''
              }${isUnavailable ? ' criterion__option--unavailable' : ''}`}
            >
              <input
                type="radio"
                id={inputId}
                name={name}
                value={approach.key}
                checked={value === approach.key}
                onChange={() => onChange(approach.key)}
                disabled={disabled || isUnavailable}
                aria-invalid={error ? true : undefined}
                aria-describedby={error ? errorId : undefined}
                className="ui-visually-hidden"
              />
              <span className="criterion__marker" aria-hidden="true">
                {marker}
              </span>
              <span className="criterion__option-label">{approach.label}</span>
            </label>
          );
        })}
      </div>

      {error && <FormFieldError id={errorId} message={error} />}
    </fieldset>
  );
}
