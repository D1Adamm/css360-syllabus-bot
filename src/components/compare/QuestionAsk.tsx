import { useId, useState } from 'react';
import { Button } from '../ui/Button';

export interface QuestionAskProps {
  /** Suggested questions for this course. May be empty. */
  examples: string[];
  /** Describes where the suggestions came from, so the UI cannot overclaim. */
  examplesLabel: string;
  /** True while suggestions are being fetched for this course. */
  examplesLoading?: boolean;
  isRunning: boolean;
  onAsk: (question: string) => void;
}

const VISIBLE_EXAMPLES = 4;

/**
 * The question box.
 *
 * Free text is the primary path — that is what a student actually wants to do.
 * The predefined questions are demoted to suggestion chips underneath, which
 * fill the box and run immediately, so they are a shortcut rather than a
 * separate mode with its own dropdown and button.
 */
export function QuestionAsk({
  examples,
  examplesLabel,
  examplesLoading = false,
  isRunning,
  onAsk,
}: QuestionAskProps) {
  const inputId = useId();
  const [value, setValue] = useState('');
  const [showAll, setShowAll] = useState(false);

  const visible = showAll ? examples : examples.slice(0, VISIBLE_EXAMPLES);
  const remaining = examples.length - visible.length;

  function submit(event: React.FormEvent) {
    event.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || isRunning) {
      return;
    }
    onAsk(trimmed);
  }

  function askExample(question: string) {
    if (isRunning) {
      return;
    }
    setValue(question);
    onAsk(question);
  }

  return (
    <section className="ask" aria-label="Ask a question">
      <form className="ask__form" onSubmit={submit}>
        <label className="ask__label" htmlFor={inputId}>
          What would you like to ask about this course?
        </label>
        <div className="ask__controls">
          <input
            id={inputId}
            className="ask__input"
            type="text"
            value={value}
            onChange={(event) => setValue(event.target.value)}
            placeholder="How much of my grade comes from the final project?"
            autoComplete="off"
          />
          <Button
            type="submit"
            variant="primary"
            iconRight="forward"
            disabled={isRunning || value.trim() === ''}
            loading={isRunning}
            loadingLabel="Asking…"
          >
            Ask
          </Button>
        </div>
      </form>

      {/* Placeholders hold the row's height while suggestions load, so the
          question box does not jump when they arrive. */}
      {examplesLoading && (
        <div className="ask__examples" aria-hidden="true">
          <span className="ask__examples-label">&nbsp;</span>
          <ul className="ask__chips">
            {[14, 18, 12].map((width, index) => (
              <li key={index}>
                <span className="ask__chip ask__chip--skeleton" style={{ width: `${width}ch` }} />
              </li>
            ))}
          </ul>
        </div>
      )}

      {!examplesLoading && examples.length > 0 && (
        <div className="ask__examples">
          <span className="ask__examples-label" id={`${inputId}-examples`}>
            {examplesLabel}
          </span>
          <ul className="ask__chips" aria-labelledby={`${inputId}-examples`}>
            {visible.map((example) => (
              <li key={example}>
                <button
                  type="button"
                  className="ask__chip"
                  onClick={() => askExample(example)}
                  disabled={isRunning}
                >
                  {example}
                </button>
              </li>
            ))}
            {remaining > 0 && (
              <li>
                <button
                  type="button"
                  className="ask__chip ask__chip--more"
                  onClick={() => setShowAll(true)}
                  disabled={isRunning}
                >
                  +{remaining} more
                </button>
              </li>
            )}
          </ul>
        </div>
      )}
    </section>
  );
}
