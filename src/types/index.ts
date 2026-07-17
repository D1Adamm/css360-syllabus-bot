export type SeedDifficulty = 'Easy' | 'Medium' | 'Hard';

export type SeedOrigin = 'prototype' | 'user' | 'ai_generated';

export type ModelKey =
  | 'base'
  | 'rag'
  | 'fineTuned'
  | 'fineTunedRag';

export interface SeedValidationComponents {
  grounded: number;
  correct: number;
  clear: number;
  useful: number;
  naturalStudentWording: number;
  categoryCorrect: number;
  notTrivialOrTemporary: number;
}

export interface SeedValidationInfo {
  score: number;
  reason: string;
  unsupportedClaims?: string[];
  components?: SeedValidationComponents;
  /** Legacy boolean or numeric mirrors from older records. */
  grounded?: boolean | number;
  correct?: boolean | number;
  clear?: boolean | number;
  useful?: boolean | number;
}

export interface SeedExample {
  id: string;
  instruction: string;
  response: string;
  category: string;
  sourceSection: string;
  difficulty: SeedDifficulty;
  directlyAnswered: boolean;
  origin: SeedOrigin;
  notes?: string;
  createdAt?: string;
  status?: string;
  questionType?: string;
  sourceChunkIds?: string[];
  validation?: SeedValidationInfo;
}

export interface ComparisonResponse {
  text: string;
  grounding: 'Low' | 'Medium' | 'High';
  simulated: boolean;
}

export interface ComparisonRecord {
  id: string;
  question: string;
  category: string;
  relevantSyllabusSection: string;
  baseResponse: ComparisonResponse;
  ragResponse: ComparisonResponse;
  fineTunedResponse: ComparisonResponse;
  fineTunedRagResponse: ComparisonResponse;
  notes: string;
}

export interface EvaluationRecord {
  id: string;
  comparisonId: string;
  mostAccurate: ModelKey;
  mostHelpful: ModelKey;
  mostConcise: ModelKey;
  bestGrounded: ModelKey;
  preferredModel: ModelKey;
  hallucinationFlags: ModelKey[];
  comment?: string;
  createdAt: string;
}

/** Syllabus processing status for a course in Firebase. */
export type SyllabusStatus =
  | 'none'
  | 'not_uploaded'
  | 'uploaded'
  | 'extracted'
  | 'indexed'
  | 'upload_failed'
  | 'index_failed'
  | 'processing'
  | 'ready'
  | 'error';

/**
 * Metadata stored at courses/{courseId}/metadata.
 * File-related fields are nullable until a syllabus is uploaded.
 */
export interface CourseMetadata {
  name: string;
  title: string;
  term: string;
  instructorName: string;
  createdAt: string;
  syllabusStatus: SyllabusStatus;
  syllabusFileName: string | null;
  syllabusType: string | null;
  chunkCount: number;
}
