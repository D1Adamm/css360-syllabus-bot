import { useEffect, useState } from 'react';
import { listCourseSeeds } from '../lib/api';
import { exampleQuestion, resolveExampleStatus } from '../lib/exampleCounts';

/**
 * Suggested questions for one course's Compare page.
 *
 * These used to come from `data/comparisonData.json` — a fixed list written for
 * CSS 360. Several of its entries assume that course's specific practices
 * ("What do students do during standup?", "Can grade questions be discussed in
 * Discord?", "Can the final reflection receive the normal extension?"), so on
 * any other course they were presented as that course's questions while being
 * about something else entirely. On an Open Source Studio syllabus, half of
 * them are unanswerable.
 *
 * The honest source is the course's own approved examples: written against that
 * syllabus and checked by that instructor. When a course has none yet, we fall
 * back to questions that are genuinely true of any course rather than
 * borrowing another course's.
 */

/** Safe for any syllabus. No course-specific practice is assumed. */
export const GENERIC_SUGGESTIONS = [
  'Is there a required textbook?',
  'How should I contact the instructor?',
  'When are assignments normally due?',
  'What is the late work policy?',
  'Are there exams in this course?',
] as const;

export type SuggestionSource = 'course' | 'generic';

export interface QuestionSuggestions {
  questions: string[];
  /** Lets the UI say where these came from instead of implying provenance. */
  source: SuggestionSource;
  loading: boolean;
}

const MAX_SUGGESTIONS = 6;

/** Approved and approved-after-editing examples are the reviewed ones. */
function isReviewedApproved(status: string): boolean {
  return status === 'approved' || status === 'edited';
}

export function useQuestionSuggestions(courseId: string): QuestionSuggestions {
  const [state, setState] = useState<QuestionSuggestions>({
    questions: [...GENERIC_SUGGESTIONS],
    source: 'generic',
    loading: true,
  });

  useEffect(() => {
    let cancelled = false;

    setState({
      questions: [...GENERIC_SUGGESTIONS],
      source: 'generic',
      loading: true,
    });

    void listCourseSeeds(courseId)
      .then((response) => {
        if (cancelled) {
          return;
        }

        const seen = new Set<string>();
        const questions: string[] = [];

        for (const seed of response.seeds ?? []) {
          if (!isReviewedApproved(resolveExampleStatus(seed))) {
            continue;
          }
          const question = exampleQuestion(seed);
          const key = question.toLowerCase();
          if (!question || seen.has(key)) {
            continue;
          }
          seen.add(key);
          questions.push(question);
          if (questions.length >= MAX_SUGGESTIONS) {
            break;
          }
        }

        setState(
          questions.length > 0
            ? { questions, source: 'course', loading: false }
            : { questions: [...GENERIC_SUGGESTIONS], source: 'generic', loading: false },
        );
      })
      .catch(() => {
        // Suggestions are a convenience. A failure here must never block the
        // question box, so fall back quietly rather than surfacing an error.
        if (!cancelled) {
          setState({
            questions: [...GENERIC_SUGGESTIONS],
            source: 'generic',
            loading: false,
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [courseId]);

  return state;
}
