import type {
  SeedDifficulty,
  SeedExample,
  SeedOrigin,
  SeedReviewStatus,
  SeedValidationInfo,
} from '../types';

export const SEED_CATEGORIES = [
  'Course Basics',
  'Communication',
  'Attendance',
  'Course Preparation',
  'Assignments',
  'Projects',
  'Standups',
  'Case Discussions',
  'Grading',
  'Late Work',
  'AI Policy',
  'Technology',
  'Office Hours',
  'Exams and Quizzes',
  'Course Expectations',
] as const;

export const ALL_CATEGORIES = 'All categories';
export const ALL_DIFFICULTIES = 'All difficulties';
export const ALL_ANSWER_TYPES = 'All';
export const ALL_REVIEW_STATUSES = 'all';

export type AnswerTypeFilter = typeof ALL_ANSWER_TYPES | 'Directly answered' | 'Not directly answered';
export type ReviewStatusFilter = typeof ALL_REVIEW_STATUSES | SeedReviewStatus;

export const REVIEW_STATUS_FILTERS: { id: ReviewStatusFilter; label: string }[] = [
  { id: 'approved', label: 'Approved' },
  { id: 'generated', label: 'Generated' },
  { id: 'rejected', label: 'Rejected' },
  { id: 'edited', label: 'Edited' },
  { id: 'all', label: 'All' },
];

export type SortOption =
  | 'id-asc'
  | 'category-asc'
  | 'difficulty'
  | 'question-asc';

const DIFFICULTY_ORDER: Record<SeedDifficulty, number> = {
  Easy: 0,
  Medium: 1,
  Hard: 2,
};

const VALID_DIFFICULTIES: readonly SeedDifficulty[] = ['Easy', 'Medium', 'Hard'];
const VALID_ORIGINS: readonly SeedOrigin[] = ['prototype', 'user', 'ai_generated'];

export function normalizeSearchText(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, ' ');
}

export function normalizeQuestion(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .replace(/\s+/g, ' ');
}

export function isMeaningfulText(value: string): boolean {
  const lettersOrDigits = value.replace(/[^a-zA-Z0-9]/g, '');
  return lettersOrDigits.length >= 3;
}

export function generateUserSeedId(): string {
  const timestamp = Date.now();
  const random = Math.random().toString(36).slice(2, 8);
  return `user-seed-${timestamp}-${random}`;
}

export function getSeedOriginLabel(origin: SeedOrigin): string {
  switch (origin) {
    case 'user':
      return 'User created';
    case 'ai_generated':
      return 'AI-generated starter seed';
    case 'prototype':
    default:
      return 'Prototype generated';
  }
}

export function isDuplicateQuestion(
  question: string,
  existingSeeds: SeedExample[],
): boolean {
  const normalized = normalizeQuestion(question);
  return existingSeeds.some(
    (seed) => normalizeQuestion(seed.instruction) === normalized,
  );
}

export function getUniqueSourceSections(seeds: SeedExample[]): string[] {
  const sections = new Set(seeds.map((seed) => seed.sourceSection));
  return Array.from(sections).sort((left, right) => left.localeCompare(right));
}

function isSeedDifficulty(value: unknown): value is SeedDifficulty {
  return typeof value === 'string' && VALID_DIFFICULTIES.includes(value as SeedDifficulty);
}

function isSeedOrigin(value: unknown): value is SeedOrigin {
  return typeof value === 'string' && VALID_ORIGINS.includes(value as SeedOrigin);
}

function readNonEmptyString(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null;
  }
  const trimmed = value.trim();
  return trimmed === '' ? null : trimmed;
}

function readFiniteScore(value: unknown): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return null;
  }
  if (value < 0 || value > 1) {
    return null;
  }
  return value;
}

function parseUnsupportedClaims(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const claims = value
    .map((item) => (typeof item === 'string' ? item.trim() : ''))
    .filter((item) => item.length > 0);
  return claims;
}

function parseValidationComponents(
  value: unknown,
): SeedValidationInfo['components'] | undefined {
  if (!value || typeof value !== 'object') {
    return undefined;
  }
  const record = value as Record<string, unknown>;
  const grounded = readFiniteScore(record.grounded);
  const correct = readFiniteScore(record.correct);
  const clear = readFiniteScore(record.clear);
  const useful = readFiniteScore(record.useful);
  const naturalStudentWording = readFiniteScore(record.naturalStudentWording);
  const categoryCorrect = readFiniteScore(record.categoryCorrect);
  const notTrivialOrTemporary = readFiniteScore(record.notTrivialOrTemporary);
  if (
    grounded === null ||
    correct === null ||
    clear === null ||
    useful === null ||
    naturalStudentWording === null ||
    categoryCorrect === null ||
    notTrivialOrTemporary === null
  ) {
    return undefined;
  }
  return {
    grounded,
    correct,
    clear,
    useful,
    naturalStudentWording,
    categoryCorrect,
    notTrivialOrTemporary,
  };
}

function parseValidation(value: unknown): SeedValidationInfo | undefined {
  if (!value || typeof value !== 'object') {
    return undefined;
  }

  const record = value as Record<string, unknown>;
  const score = readFiniteScore(record.score);
  const reason = readNonEmptyString(record.reason);
  if (score === null || reason === null) {
    return undefined;
  }

  const components = parseValidationComponents(record.components);
  const unsupportedClaims = parseUnsupportedClaims(record.unsupportedClaims);

  // Legacy boolean-only records (pre-rubric).
  const legacyBooleans =
    typeof record.grounded === 'boolean' &&
    typeof record.correct === 'boolean' &&
    typeof record.clear === 'boolean' &&
    typeof record.useful === 'boolean';

  if (!components && !legacyBooleans) {
    // Accept numeric top-level components without nested components object.
    const topLevelComponents = parseValidationComponents(record);
    if (!topLevelComponents) {
      return undefined;
    }
    return {
      score,
      reason,
      components: topLevelComponents,
      ...(unsupportedClaims ? { unsupportedClaims } : {}),
      grounded: topLevelComponents.grounded,
      correct: topLevelComponents.correct,
      clear: topLevelComponents.clear,
      useful: topLevelComponents.useful,
    };
  }

  const validation: SeedValidationInfo = {
    score,
    reason,
  };

  if (components) {
    validation.components = components;
  }
  if (unsupportedClaims && unsupportedClaims.length > 0) {
    validation.unsupportedClaims = unsupportedClaims;
  } else if (unsupportedClaims) {
    validation.unsupportedClaims = [];
  }

  if (legacyBooleans) {
    validation.grounded = record.grounded as boolean;
    validation.correct = record.correct as boolean;
    validation.clear = record.clear as boolean;
    validation.useful = record.useful as boolean;
  } else if (components) {
    validation.grounded = components.grounded;
    validation.correct = components.correct;
    validation.clear = components.clear;
    validation.useful = components.useful;
  }

  return validation;
}

function parseSourceChunkIds(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const ids = value
    .map((item) => (typeof item === 'string' ? item.trim() : ''))
    .filter((item) => item.length > 0);
  return ids.length > 0 ? ids : undefined;
}

/**
 * Normalize a stored (or local fixture) seed record into the SeedExample shape.
 * Accepts dual AI field names (question/answer) and classic instruction/response.
 */
export function normalizeSeedExample(
  value: unknown,
  fallbackId?: string,
): SeedExample | null {
  if (!value || typeof value !== 'object') {
    return null;
  }

  const record = value as Record<string, unknown>;
  const origin = record.origin;
  if (!isSeedOrigin(origin)) {
    return null;
  }

  const id = readNonEmptyString(record.id) ?? readNonEmptyString(fallbackId);
  const instruction =
    readNonEmptyString(record.instruction) ?? readNonEmptyString(record.question);
  const response =
    readNonEmptyString(record.response) ?? readNonEmptyString(record.answer);

  if (!id || !instruction || !response) {
    return null;
  }

  const sourceChunkIds = parseSourceChunkIds(record.sourceChunkIds);
  const sourceSection =
    readNonEmptyString(record.sourceSection) ??
    (sourceChunkIds ? sourceChunkIds.join(', ') : null) ??
    (origin === 'ai_generated' ? 'General' : null);

  if (!sourceSection) {
    return null;
  }

  let difficulty: SeedDifficulty;
  if (isSeedDifficulty(record.difficulty)) {
    difficulty = record.difficulty;
  } else if (origin === 'ai_generated') {
    difficulty = 'Medium';
  } else {
    return null;
  }

  let directlyAnswered: boolean;
  if (typeof record.directlyAnswered === 'boolean') {
    directlyAnswered = record.directlyAnswered;
  } else if (origin === 'ai_generated') {
    directlyAnswered = true;
  } else {
    return null;
  }

  const category = readNonEmptyString(record.category);
  if (!category) {
    return null;
  }

  const seed: SeedExample = {
    id,
    instruction,
    response,
    category,
    sourceSection,
    difficulty,
    directlyAnswered,
    origin,
  };

  const notes = readNonEmptyString(record.notes);
  if (notes) {
    seed.notes = notes;
  }

  const createdAt = readNonEmptyString(record.createdAt);
  if (createdAt) {
    seed.createdAt = createdAt;
  }

  const status = readNonEmptyString(record.status);
  if (status) {
    seed.status = status;
  }

  const reviewStatus = readNonEmptyString(record.reviewStatus);
  if (reviewStatus) {
    seed.reviewStatus = reviewStatus;
  }

  const reviewNotes = readNonEmptyString(record.reviewNotes);
  if (reviewNotes) {
    seed.reviewNotes = reviewNotes;
  }

  const originalQuestion = readNonEmptyString(record.originalQuestion);
  if (originalQuestion) {
    seed.originalQuestion = originalQuestion;
  }

  const originalAnswer = readNonEmptyString(record.originalAnswer);
  if (originalAnswer) {
    seed.originalAnswer = originalAnswer;
  }

  if (record.wasEdited === true) {
    seed.wasEdited = true;
  }

  const questionType = readNonEmptyString(record.questionType);
  if (questionType) {
    seed.questionType = questionType;
  }

  if (sourceChunkIds) {
    seed.sourceChunkIds = sourceChunkIds;
  }

  const validation = parseValidation(record.validation);
  if (validation) {
    seed.validation = validation;
  }

  return seed;
}

export function isSeedExample(value: unknown): value is SeedExample {
  return normalizeSeedExample(value) !== null;
}

export function isSeedExampleArray(value: unknown): value is SeedExample[] {
  if (!Array.isArray(value)) {
    return false;
  }

  return value.every(isSeedExample);
}

export function seedMatchesSearch(seed: SeedExample, query: string): boolean {
  if (!query) {
    return true;
  }

  const searchableText = [
    seed.instruction,
    seed.response,
    seed.category,
    seed.sourceSection,
  ]
    .join(' ')
    .toLowerCase();

  return searchableText.includes(query);
}

export function filterByCategory(
  seeds: SeedExample[],
  category: string,
): SeedExample[] {
  if (category === ALL_CATEGORIES) {
    return seeds;
  }

  return seeds.filter((seed) => seed.category === category);
}

export function filterByDifficulty(
  seeds: SeedExample[],
  difficulty: string,
): SeedExample[] {
  if (difficulty === ALL_DIFFICULTIES) {
    return seeds;
  }

  return seeds.filter((seed) => seed.difficulty === difficulty);
}

export function filterByAnswerType(
  seeds: SeedExample[],
  answerType: AnswerTypeFilter,
): SeedExample[] {
  if (answerType === ALL_ANSWER_TYPES) {
    return seeds;
  }

  if (answerType === 'Directly answered') {
    return seeds.filter((seed) => seed.directlyAnswered);
  }

  return seeds.filter((seed) => !seed.directlyAnswered);
}

export function sortSeeds(seeds: SeedExample[], sortBy: SortOption): SeedExample[] {
  const sorted = [...seeds];

  switch (sortBy) {
    case 'id-asc':
      return sorted.sort((left, right) => left.id.localeCompare(right.id));
    case 'category-asc':
      return sorted.sort((left, right) => {
        const categoryCompare = left.category.localeCompare(right.category);
        return categoryCompare !== 0 ? categoryCompare : left.id.localeCompare(right.id);
      });
    case 'difficulty':
      return sorted.sort((left, right) => {
        const difficultyCompare =
          DIFFICULTY_ORDER[left.difficulty] - DIFFICULTY_ORDER[right.difficulty];
        return difficultyCompare !== 0 ? difficultyCompare : left.id.localeCompare(right.id);
      });
    case 'question-asc':
      return sorted.sort((left, right) =>
        left.instruction.localeCompare(right.instruction),
      );
    default:
      return sorted;
  }
}

export interface DatasetStatistics {
  totalExamples: number;
  approvedCount: number;
  rejectedCount: number;
  generatedCount: number;
  editedCount: number;
  totalCategories: number;
  easyCount: number;
  mediumCount: number;
  hardCount: number;
  directlyAnsweredCount: number;
  notDirectlyAnsweredCount: number;
}

export function resolveSeedReviewStatus(seed: SeedExample): SeedReviewStatus | string {
  const raw = String(seed.reviewStatus || seed.status || 'generated')
    .trim()
    .toLowerCase();
  if (
    raw === 'generated' ||
    raw === 'approved' ||
    raw === 'rejected' ||
    raw === 'edited'
  ) {
    return raw;
  }
  return 'generated';
}

export function countByReviewStatus(seeds: SeedExample[]): Record<SeedReviewStatus, number> {
  const counts: Record<SeedReviewStatus, number> = {
    generated: 0,
    approved: 0,
    rejected: 0,
    edited: 0,
  };
  for (const seed of seeds) {
    const status = resolveSeedReviewStatus(seed);
    if (status in counts) {
      counts[status as SeedReviewStatus] += 1;
    } else {
      counts.generated += 1;
    }
  }
  return counts;
}

export function calculateStatistics(seeds: SeedExample[]): DatasetStatistics {
  const categories = new Set(seeds.map((seed) => seed.category));
  const byReview = countByReviewStatus(seeds);

  return {
    totalExamples: seeds.length,
    approvedCount: byReview.approved,
    rejectedCount: byReview.rejected,
    generatedCount: byReview.generated,
    editedCount: byReview.edited,
    totalCategories: categories.size,
    easyCount: seeds.filter((seed) => seed.difficulty === 'Easy').length,
    mediumCount: seeds.filter((seed) => seed.difficulty === 'Medium').length,
    hardCount: seeds.filter((seed) => seed.difficulty === 'Hard').length,
    directlyAnsweredCount: seeds.filter((seed) => seed.directlyAnswered).length,
    notDirectlyAnsweredCount: seeds.filter((seed) => !seed.directlyAnswered).length,
  };
}

export function getUniqueCategories(seeds: SeedExample[]): string[] {
  const categories = new Set(seeds.map((seed) => seed.category));
  return Array.from(categories).sort((left, right) => left.localeCompare(right));
}

export function filterSeeds(
  seeds: SeedExample[],
  options: {
    searchQuery: string;
    category: string;
    difficulty: string;
    answerType: AnswerTypeFilter;
    reviewStatus?: ReviewStatusFilter;
    sortBy: SortOption;
  },
): SeedExample[] {
  const normalizedQuery = normalizeSearchText(options.searchQuery);
  const reviewFilter = options.reviewStatus ?? ALL_REVIEW_STATUSES;

  const filtered = seeds.filter((seed) => {
    const matchesSearch = seedMatchesSearch(seed, normalizedQuery);
    const matchesCategory =
      options.category === ALL_CATEGORIES || seed.category === options.category;
    const matchesDifficulty =
      options.difficulty === ALL_DIFFICULTIES || seed.difficulty === options.difficulty;
    const matchesAnswerType =
      options.answerType === ALL_ANSWER_TYPES ||
      (options.answerType === 'Directly answered' && seed.directlyAnswered) ||
      (options.answerType === 'Not directly answered' && !seed.directlyAnswered);
    const matchesReview =
      reviewFilter === ALL_REVIEW_STATUSES ||
      resolveSeedReviewStatus(seed) === reviewFilter;

    return (
      matchesSearch &&
      matchesCategory &&
      matchesDifficulty &&
      matchesAnswerType &&
      matchesReview
    );
  });

  return sortSeeds(filtered, options.sortBy);
}
