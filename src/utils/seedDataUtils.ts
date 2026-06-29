import type { SeedDifficulty, SeedExample } from '../types';

export const USER_SEEDS_STORAGE_KEY = 'syllabus-demo-user-seeds';

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

export type AnswerTypeFilter = typeof ALL_ANSWER_TYPES | 'Directly answered' | 'Not directly answered';

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

export function combinePrototypeAndUserSeeds(
  prototypeSeeds: SeedExample[],
  userSeeds: SeedExample[],
): SeedExample[] {
  return [...prototypeSeeds, ...userSeeds];
}

export function isSeedExampleArray(value: unknown): value is SeedExample[] {
  if (!Array.isArray(value)) {
    return false;
  }

  return value.every(
    (item) =>
      typeof item === 'object' &&
      item !== null &&
      typeof (item as SeedExample).id === 'string' &&
      typeof (item as SeedExample).instruction === 'string' &&
      typeof (item as SeedExample).response === 'string' &&
      (item as SeedExample).origin === 'user',
  );
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
  totalCategories: number;
  easyCount: number;
  mediumCount: number;
  hardCount: number;
  directlyAnsweredCount: number;
  notDirectlyAnsweredCount: number;
}

export function calculateStatistics(seeds: SeedExample[]): DatasetStatistics {
  const categories = new Set(seeds.map((seed) => seed.category));

  return {
    totalExamples: seeds.length,
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
    sortBy: SortOption;
  },
): SeedExample[] {
  const normalizedQuery = normalizeSearchText(options.searchQuery);

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

    return matchesSearch && matchesCategory && matchesDifficulty && matchesAnswerType;
  });

  return sortSeeds(filtered, options.sortBy);
}
