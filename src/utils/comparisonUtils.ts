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

export interface MatchResult {
  recordId: string;
  matchedQuestion: string;
  score: number;
}

const MIN_MATCH_SCORE = 0.25;

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
    const overlap = queryTokens.filter((token) => recordTokens.has(token)).length;
    const score = overlap / queryTokens.length;

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
