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

/* ------------------------------------------------------------------------ *
 * Course model registry
 *
 * Stored at `courses/{courseId}/model`, alongside the course's other durable
 * records. Two things are deliberately kept apart:
 *
 *   - `status` — does a trained model exist for this course, and is it usable?
 *     This is a durable fact about an artifact that was produced. It stays true
 *     whether or not anything is currently serving it.
 *   - `deployment` — is that model actually loaded somewhere answering
 *     requests right now? This changes when a service starts or stops and says
 *     nothing about whether the model exists.
 *
 * Conflating them is what made the UI unable to describe a trained-but-offline
 * model, which is exactly CSS 360's situation.
 * ------------------------------------------------------------------------ */

/** Training/artifact state. Durable; survives the service going away. */
export type CourseModelStatus =
  /** Training completed and the artifact is registered. */
  | 'ready'
  /** Training is running. Set by whoever runs it; nothing infers this. */
  | 'training'
  /** Training was attempted and did not produce a usable artifact. */
  | 'failed';

/** Whether the registered model is currently being served. */
export type CourseModelDeploymentStatus =
  /** Loaded and serving requests. */
  | 'online'
  /** Registered but nothing is serving it. */
  | 'offline'
  /** Not recorded. Never guessed from service health. */
  | 'unknown';

export interface CourseModelVersion {
  /** `v1`, `v2`, … Monotonic per course. */
  version: string;
  /** Model the adapter was trained on, e.g. `meta-llama/Llama-3.2-3B-Instruct`. */
  baseModel: string;
  /** Approved examples the version was trained from. */
  trainingExampleCount: number;
  status: CourseModelStatus;
  deployment: CourseModelDeploymentStatus;
  /**
   * Logical artifact reference, e.g. `css-360-qlora/adapter`.
   *
   * Deliberately relative and machine-independent: the absolute path embeds a
   * cluster home directory and a username. Admin surfaces may show this;
   * professor surfaces never do.
   */
  artifactRef: string;
  createdAt: string;
  updatedAt?: string;
  notes?: string;
}

export interface CourseModelRegistry {
  /** Version key that Professor-facing screens describe. */
  currentVersion: string;
  versions: Record<string, CourseModelVersion>;
}

/* ------------------------------------------------------------------------ *
 * Course model requests
 *
 * Stored at `courses/{courseId}/modelRequest`, deliberately outside the model
 * registry. The registry describes artifacts that exist; a request describes
 * work that has been asked for and has not produced one yet. A course can have
 * a request and no model, a model and no request, or both.
 * ------------------------------------------------------------------------ */

export type CourseModelRequestStatus =
  /** Submitted by a professor; nobody has picked it up. */
  | 'requested'
  /** Being set up — dataset export, split, queueing. */
  | 'preparing'
  /** A training run is under way. */
  | 'training'
  /** Finished; a model was registered. Terminal. */
  | 'ready'
  /** Did not produce a usable model. Terminal. */
  | 'failed';

export interface CourseModelRequest {
  courseId: string;
  status: CourseModelRequestStatus;
  requestedAt: string;
  updatedAt: string;
  /** Approved examples at the moment of the request, for later comparison. */
  approvedExampleCount: number;
  /** Set only when `status` is `failed`. Never shown to a professor verbatim. */
  failureMessage?: string;
}
