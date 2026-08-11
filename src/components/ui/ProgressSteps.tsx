import { Icon } from './Icon';

export type StepStatus = 'complete' | 'current' | 'upcoming';

export interface ProgressStep {
  id: string;
  label: string;
  /** A short factual line, e.g. "3 questions". Not a sentence. */
  meta?: string;
}

export interface ProgressStepsProps {
  steps: ProgressStep[];
  /** Index of the step the user is on. Earlier steps render as complete. */
  currentIndex: number;
  /** Marks the whole sequence finished, so the last step reads as complete. */
  allComplete?: boolean;
  /**
   * Explicit status per step, overriding `currentIndex`. Use this where the
   * component orients the reader rather than reporting their history — there
   * is no honest way to mark a step "complete" without knowing who is looking.
   */
  statuses?: StepStatus[];
  className?: string;
  'aria-label'?: string;
}

function statusFor(
  index: number,
  currentIndex: number,
  allComplete: boolean,
): StepStatus {
  if (allComplete || index < currentIndex) {
    return 'complete';
  }
  return index === currentIndex ? 'current' : 'upcoming';
}

/**
 * The Contribute -> Compare -> Evaluate spine, shown as a lightweight
 * orientation device rather than a dashboard.
 *
 * Rendered as an ordered list so it reads correctly without styles, with
 * `aria-current="step"` on the active item.
 */
export function ProgressSteps({
  steps,
  currentIndex,
  allComplete = false,
  statuses,
  className,
  'aria-label': ariaLabel = 'Your progress',
}: ProgressStepsProps) {
  const classes = ['ui-steps', className].filter(Boolean).join(' ');

  return (
    <ol className={classes} aria-label={ariaLabel}>
      {steps.map((step, index) => {
        const status = statuses?.[index] ?? statusFor(index, currentIndex, allComplete);

        return (
          <li
            key={step.id}
            className={`ui-steps__item ui-steps__item--${status}`}
            aria-current={status === 'current' ? 'step' : undefined}
          >
            <span className="ui-steps__marker" aria-hidden="true">
              {status === 'complete' ? <Icon name="success" size={12} /> : index + 1}
            </span>
            <span className="ui-steps__text">
              <span className="ui-steps__label">{step.label}</span>
              {step.meta && <span className="ui-steps__meta">{step.meta}</span>}
            </span>
            <span className="ui-visually-hidden">
              {status === 'complete'
                ? ' (completed)'
                : status === 'current'
                  ? ' (current step)'
                  : ' (not started)'}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
