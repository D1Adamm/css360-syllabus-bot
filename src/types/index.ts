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

/** Syllabus processing status for a course. */
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
  /**
   * Automatic starter-seed generation, written by the backend job.
   *
   * Optional because courses created before the job existed have no record,
   * and because a course whose syllabus was never indexed never starts one.
   */
  starterSeedGeneration?: StoredStarterSeedGeneration;
}

/**
 * The record exactly as the generation job stores it.
 *
 * Its vocabulary is the job's, not the professor's: `queued` and `generating`
 * are two points in one wait, and `partial` means fewer examples than asked for
 * were produced — which is still a course with examples to review. Translating
 * that for a professor is `starterSeedGeneration.ts`'s job, not this type's.
 *
 * `error` holds whatever the model or backend said. It is operator detail and
 * never reaches a professor.
 */
export interface StoredStarterSeedGeneration {
  status?: string;
  targetCount?: number;
  finalCount?: number;
  savedCount?: number;
  failedToSaveCount?: number;
  error?: string | null;
  startedAt?: string | null;
  completedAt?: string | null;
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
  /**
   * Training-data preparation, recorded once an administrator has run it.
   *
   * Metadata only — the dataset itself lives on the backend under
   * `data/exports/{courseId}/`. `datasetRef` is deliberately relative: absolute
   * paths embed a machine layout and must not reach anything a professor can
   * read.
   */
  preparation?: CourseModelRequestPreparation;
  /**
   * Why the last preparation attempt failed. Admin-only, and cleared on the
   * next success. The request stays `requested` so it can simply be retried.
   */
  preparationError?: string;
  /** Submitted training job. Admin-only; no field here reaches a professor. */
  training?: CourseModelRequestTraining;
  /**
   * Why the last launch attempt failed. Admin-only, cleared on success. The
   * request stays `preparing` so it can be retried.
   */
  launchError?: string;
  /**
   * The training run currently carrying this request, if one has been queued.
   *
   * A pointer and nothing else. Operational state — claims, attempts, job
   * identifiers — lives on the run itself at
   * `courses/{courseId}/trainingRuns/{runId}`, so a professor-facing record
   * never has to carry it.
   */
  currentRunId?: string;
}

export interface CourseModelRequestTraining {
  /** Slurm job id, captured from the real submission. */
  jobId: string;
  /** `smoke` or `full`. */
  mode: string;
  submittedAt: string;
  /** Relative, course-scoped dataset the job was given. */
  datasetRef: string;
  trainExamples: number;
  validationExamples: number;
}

export interface CourseModelRequestPreparation {
  preparedAt: string;
  /** Approved examples actually exported, re-counted at preparation time. */
  sourceApprovedExampleCount: number;
  /** Relative, course-scoped, e.g. `exports/css-490-spring-2026-cgvl`. */
  datasetRef: string;
  trainExamples: number;
  validationExamples: number;
  /** Recorded so a split can be reproduced exactly. */
  splitSeed?: number;
}

/* ------------------------------------------------------------------------ *
 * Training runs
 *
 * Stored at `courses/{courseId}/trainingRuns/{runId}`: the durable queue a
 * runner on the cluster reads. Operational only. It is kept apart from
 * `modelRequest` because the two answer different questions — a professor asks
 * "is my model coming?", an operator asks "what is queued, who holds it, and
 * how many times has it been tried?" — and because a professor-facing record
 * must never grow fields that describe infrastructure.
 * ------------------------------------------------------------------------ */

export type TrainingRunState =
  /** Enqueued by an administrator; no runner holds it. */
  | 'queued'
  /** A runner holds a lease on it and is working on it. */
  | 'claimed'
  /** Handed to the scheduler; a job identifier exists. */
  | 'submitted'
  /** The job is running. */
  | 'training'
  /** Finished and produced what was asked for. Terminal. */
  | 'succeeded'
  /** Did not produce a usable result. Terminal. */
  | 'failed';

/**
 * A time-limited hold by exactly one runner.
 *
 * The lease is what stops two runners doing the same work. It expires so that a
 * runner which dies mid-run — a dropped session, a rebooted login node — cannot
 * strand a run forever.
 */
export interface TrainingRunClaim {
  /** Who holds it, for operators reading the queue. */
  owner: string;
  claimedAt: string;
  expiresAt: string;
}

export interface TrainingRun {
  runId: string;
  courseId: string;
  /** `smoke` or `full`. Smoke is never chained into full automatically. */
  mode: TrainingMode;
  state: TrainingRunState;
  enqueuedAt: string;
  updatedAt: string;
  /** Relative, course-scoped dataset reference the preparation stage recorded. */
  datasetRef: string;
  /** Approved examples the prepared dataset was built from. */
  approvedExampleCount: number;
  trainExamples: number;
  validationExamples: number;
  /** How many times a runner has taken this run. Starts at 0. */
  attempt: number;
  /**
   * Slurm job id, only after a real submission. Never invented by the browser.
   */
  jobId?: string;
  /** Present only while a runner holds it. */
  claim?: TrainingRunClaim;
  /** Why the last attempt failed. Operator-facing. */
  error?: string;
}

export type TrainingMode = 'smoke' | 'full';
