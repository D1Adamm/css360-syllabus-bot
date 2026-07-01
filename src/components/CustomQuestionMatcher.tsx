import { useState } from 'react';
import { findBestComparisonMatch } from '../utils/comparisonUtils';

interface CustomQuestionMatcherProps {
  records: { id: string; question: string }[];
  isSubmitting?: boolean;
  onMatch: (recordId: string, matchedQuestion: string) => void;
  onNoMatch: () => void;
  onQuestionSubmit: (question: string) => void;
}

export function CustomQuestionMatcher({
  records,
  isSubmitting = false,
  onMatch,
  onNoMatch,
  onQuestionSubmit,
}: CustomQuestionMatcherProps) {
  const [customQuestion, setCustomQuestion] = useState('');
  const [notice, setNotice] = useState<{
    type: 'match' | 'no-match';
    message: string;
  } | null>(null);

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmed = customQuestion.trim();
    if (!trimmed || isSubmitting) {
      return;
    }

    onQuestionSubmit(trimmed);

    const match = findBestComparisonMatch(trimmed, records);

    if (match) {
      onMatch(match.recordId, match.matchedQuestion);
      setNotice({
        type: 'match',
        message: `Matched to predefined question: "${match.matchedQuestion}". The Base Model and RAG answers are live; Fine-Tuned and Fine-Tuned + RAG remain simulated.`,
      });
      return;
    }

    onNoMatch();
    setNotice({
      type: 'no-match',
      message:
        'The Base Model and RAG answers are live. Fine-Tuned and Fine-Tuned + RAG remain simulated and only update when your question closely matches a predefined example.',
    });
  }

  return (
    <section className="custom-question-matcher" aria-labelledby="custom-question-title">
      <h2 id="custom-question-title" className="custom-question-matcher__title">
        Try a custom syllabus question
      </h2>

      <form className="custom-question-matcher__form" onSubmit={handleSubmit}>
        <label htmlFor="custom-question-input" className="custom-question-matcher__label">
          Enter a question to send to the Base Model and RAG
        </label>
        <div className="custom-question-matcher__controls">
          <input
            id="custom-question-input"
            type="text"
            className="custom-question-matcher__input"
            value={customQuestion}
            onChange={(event) => {
              setCustomQuestion(event.target.value);
              setNotice(null);
            }}
            placeholder="e.g., Can I email about my grade?"
          />
          <button
            type="submit"
            className="custom-question-matcher__submit"
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Generating...' : 'Ask question'}
          </button>
        </div>
      </form>

      {notice && (
        <p
          className={`custom-question-matcher__notice custom-question-matcher__notice--${notice.type}`}
          role="status"
        >
          {notice.message}
        </p>
      )}
    </section>
  );
}
