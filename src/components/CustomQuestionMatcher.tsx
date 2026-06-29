import { useState } from 'react';
import { findBestComparisonMatch } from '../utils/comparisonUtils';

interface CustomQuestionMatcherProps {
  records: { id: string; question: string }[];
  onMatch: (recordId: string, matchedQuestion: string) => void;
  onNoMatch: () => void;
}

export function CustomQuestionMatcher({
  records,
  onMatch,
  onNoMatch,
}: CustomQuestionMatcherProps) {
  const [customQuestion, setCustomQuestion] = useState('');
  const [notice, setNotice] = useState<{
    type: 'match' | 'no-match';
    message: string;
  } | null>(null);

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmed = customQuestion.trim();
    if (!trimmed) {
      return;
    }

    const match = findBestComparisonMatch(trimmed, records);

    if (match) {
      onMatch(match.recordId, match.matchedQuestion);
      setNotice({
        type: 'match',
        message: `Matched to predefined question: "${match.matchedQuestion}"`,
      });
      return;
    }

    onNoMatch();
    setNotice({
      type: 'no-match',
      message:
        'Live generation is not implemented in this prototype. Your question did not closely match any predefined example, so no new model responses were created.',
    });
  }

  return (
    <section className="custom-question-matcher" aria-labelledby="custom-question-title">
      <h2 id="custom-question-title" className="custom-question-matcher__title">
        Try a custom syllabus question
      </h2>

      <form className="custom-question-matcher__form" onSubmit={handleSubmit}>
        <label htmlFor="custom-question-input" className="custom-question-matcher__label">
          Enter a question to match against predefined examples
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
          <button type="submit" className="custom-question-matcher__submit">
            Find closest match
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
