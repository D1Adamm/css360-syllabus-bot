import type { CourseSeedReviewRecord } from './api';
import { exampleQuestion } from './exampleCounts';

/**
 * Exact-duplicate questions, flagged for the professor's eye only.
 *
 * This is not a similarity detector and must not become one. Generation
 * already writes `normalizedQuestionKey` — the question lowercased with
 * punctuation stripped — so two examples that ask literally the same thing
 * share a key. Matching on that is free and cannot be wrong about what it
 * claims: it says "these two are the same words", nothing more. Anything
 * fuzzier is a judgement, and judgement is the professor's job.
 */

const NON_ALNUM = /[^a-z0-9\s]+/g;
const WHITESPACE = /\s+/g;

/** Mirrors the backend's `normalize_question_for_dedupe`. */
export function normalizeQuestionKey(question: string): string {
  return question
    .trim()
    .toLowerCase()
    .replace(NON_ALNUM, ' ')
    .replace(WHITESPACE, ' ')
    .trim();
}

function keyFor(example: CourseSeedReviewRecord): string {
  const stored = String(example.normalizedQuestionKey || '').trim();
  return stored || normalizeQuestionKey(exampleQuestion(example));
}

/**
 * Ids of every example whose question is word-for-word another's.
 *
 * All members of a repeated group are returned, not just the later ones: the
 * professor decides which copy to keep, so neither is privileged.
 */
export function findDuplicateExampleIds(
  examples: CourseSeedReviewRecord[],
): Set<string> {
  const byKey = new Map<string, string[]>();

  for (const example of examples) {
    const id = String(example.id || '').trim();
    const key = keyFor(example);
    if (!id || !key) {
      continue;
    }
    const group = byKey.get(key);
    if (group) {
      group.push(id);
    } else {
      byKey.set(key, [id]);
    }
  }

  const duplicates = new Set<string>();
  for (const group of byKey.values()) {
    if (group.length > 1) {
      for (const id of group) {
        duplicates.add(id);
      }
    }
  }
  return duplicates;
}
