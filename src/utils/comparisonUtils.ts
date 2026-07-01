const STOP_WORDS = new Set([
  'a',
  'an',
  'the',
  'is',
  'are',
  'was',
  'were',
  'be',
  'been',
  'being',
  'have',
  'has',
  'had',
  'do',
  'does',
  'did',
  'will',
  'would',
  'could',
  'should',
  'may',
  'might',
  'can',
  'to',
  'of',
  'in',
  'for',
  'on',
  'with',
  'at',
  'by',
  'from',
  'as',
  'into',
  'through',
  'during',
  'before',
  'after',
  'above',
  'below',
  'between',
  'under',
  'again',
  'further',
  'then',
  'once',
  'here',
  'there',
  'when',
  'where',
  'why',
  'how',
  'all',
  'each',
  'few',
  'more',
  'most',
  'other',
  'some',
  'such',
  'no',
  'nor',
  'not',
  'only',
  'own',
  'same',
  'so',
  'than',
  'too',
  'very',
  'just',
  'and',
  'but',
  'if',
  'or',
  'because',
  'until',
  'while',
  'about',
  'what',
  'which',
  'who',
  'whom',
  'this',
  'that',
  'these',
  'those',
  'am',
  'i',
  'me',
  'my',
  'myself',
  'we',
  'our',
  'you',
  'your',
  'he',
  'him',
  'his',
  'she',
  'her',
  'it',
  'its',
  'they',
  'them',
  'their',
]);

/** Shared syllabus words that are too broad to justify a match on their own. */
export const BROAD_MATCH_TOKENS = new Set([
  'grade',
  'grades',
  'assignment',
  'assignments',
  'class',
  'classes',
  'project',
  'projects',
]);

const MIN_MATCH_SCORE = 0.55;
const MIN_MEANINGFUL_OVERLAP = 1;
const PREFIX_MATCH_MIN_LENGTH = 5;

export function normalizeComparisonText(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function tokenizeForMatching(value: string): string[] {
  return normalizeComparisonText(value)
    .split(' ')
    .filter((token) => token.length > 1 && !STOP_WORDS.has(token));
}

export function tokensAreEquivalent(left: string, right: string): boolean {
  if (left === right) {
    return true;
  }

  if (
    left.length >= PREFIX_MATCH_MIN_LENGTH &&
    right.length >= PREFIX_MATCH_MIN_LENGTH &&
    (left.startsWith(right) || right.startsWith(left))
  ) {
    return true;
  }

  return false;
}

export function countTokenOverlap(
  queryTokens: string[],
  recordTokens: Set<string>,
): { overlapCount: number; meaningfulOverlapCount: number } {
  let overlapCount = 0;
  let meaningfulOverlapCount = 0;

  for (const queryToken of queryTokens) {
    const hasMatch = [...recordTokens].some((recordToken) =>
      tokensAreEquivalent(queryToken, recordToken),
    );

    if (!hasMatch) {
      continue;
    }

    overlapCount += 1;

    if (!BROAD_MATCH_TOKENS.has(queryToken)) {
      meaningfulOverlapCount += 1;
    }
  }

  return { overlapCount, meaningfulOverlapCount };
}

export function computeComparisonMatchScore(
  queryTokens: string[],
  recordTokens: Set<string>,
): number {
  const { overlapCount, meaningfulOverlapCount } = countTokenOverlap(
    queryTokens,
    recordTokens,
  );

  if (overlapCount === 0) {
    return 0;
  }

  if (meaningfulOverlapCount < MIN_MEANINGFUL_OVERLAP) {
    return 0;
  }

  const precision = overlapCount / recordTokens.size;
  const recall = overlapCount / queryTokens.length;
  const denominator = precision + recall;

  if (denominator === 0) {
    return 0;
  }

  return (2 * precision * recall) / denominator;
}

export interface MatchResult {
  recordId: string;
  matchedQuestion: string;
  score: number;
}

export function findBestComparisonMatch(
  customQuestion: string,
  records: { id: string; question: string }[],
): MatchResult | null {
  const queryTokens = tokenizeForMatching(customQuestion);

  if (queryTokens.length === 0) {
    return null;
  }

  let bestMatch: MatchResult | null = null;

  for (const record of records) {
    const recordTokens = new Set(tokenizeForMatching(record.question));
    const score = computeComparisonMatchScore(queryTokens, recordTokens);

    if (score >= MIN_MATCH_SCORE && (!bestMatch || score > bestMatch.score)) {
      bestMatch = {
        recordId: record.id,
        matchedQuestion: record.question,
        score,
      };
    }
  }

  return bestMatch;
}
