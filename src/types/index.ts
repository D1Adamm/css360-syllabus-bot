export type SeedDifficulty = 'Easy' | 'Medium' | 'Hard';

export type SeedOrigin = 'prototype' | 'user' | 'ai_generated';

/** Phase 8 human review status (validated AI seeds start as generated). */
export type SeedReviewStatus = 'generated' | 'approved' | 'rejected' | 'edited';

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
  /** Phase 8 review field; falls back to status when absent. */
  reviewStatus?: SeedReviewStatus | string;
  reviewNotes?: string;
  factId?: string | null;
  evidenceQuote?: string | null;
  originalQuestion?: string | null;
  originalAnswer?: string | null;
  /** True when the seed was human-edited; survives later approval. */
  wasEdited?: boolean;
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
  /**
   * Predefined comparison this rating belongs to, when the question matched
   * one. Free-text questions store a synthetic id here and carry the wording
   * in `questionText`. Required so records written before live evaluation
   * keep aggregating exactly as they did.
   */
  comparisonId: string;
  mostAccurate: ModelKey;
  mostHelpful: ModelKey;
  mostConcise: ModelKey;
  bestGrounded: ModelKey;
  preferredModel: ModelKey;
  hallucinationFlags: ModelKey[];
  comment?: string;
  createdAt: string;
  /** Links a rating to the comparison run the student actually saw. */
  runId?: string;
  /** The question as asked. Present for free-text questions. */
  questionText?: string;
  courseId?: string;
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
