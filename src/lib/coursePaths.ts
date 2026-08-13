import { assertValidCourseId } from './courseId';

export const COURSES_ROOT_PATH = 'courses';

export function getCourseMetadataPath(courseId: string): string {
  assertValidCourseId(courseId);
  return `${COURSES_ROOT_PATH}/${courseId}/metadata`;
}

export function getCourseSeedExamplesPath(courseId: string): string {
  assertValidCourseId(courseId);
  return `courses/${courseId}/seedExamples`;
}

export function getCourseSeedExamplePath(courseId: string, exampleId: string): string {
  assertValidCourseId(courseId);
  return `courses/${courseId}/seedExamples/${exampleId}`;
}

export function getCourseEvaluationsPath(courseId: string): string {
  assertValidCourseId(courseId);
  return `courses/${courseId}/evaluations`;
}

export function getCourseEvaluationPath(courseId: string, evaluationId: string): string {
  assertValidCourseId(courseId);
  return `courses/${courseId}/evaluations/${evaluationId}`;
}

export function getCourseModelPath(courseId: string): string {
  assertValidCourseId(courseId);
  return `courses/${courseId}/model`;
}

/**
 * A course's outstanding model request.
 *
 * Sibling of `model`, never inside it: the registry records artifacts that
 * exist, and a request is work that has not happened yet. Mixing them would
 * mean a pending request looked like a model.
 */
export function getCourseModelRequestPath(courseId: string): string {
  assertValidCourseId(courseId);
  return `courses/${courseId}/modelRequest`;
}

/**
 * A course's training runs — the durable queue a cluster runner reads.
 *
 * Course-scoped like everything else here, so a runner asked for one course can
 * only ever see that course's work, and operational state never has to be
 * stored on the professor-facing request.
 */
export function getCourseTrainingRunsPath(courseId: string): string {
  assertValidCourseId(courseId);
  return `courses/${courseId}/trainingRuns`;
}

export function getCourseTrainingRunPath(courseId: string, runId: string): string {
  assertValidTrainingRunId(runId);
  return `${getCourseTrainingRunsPath(courseId)}/${runId}`;
}

/** Firebase keys cannot contain `.`, `$`, `#`, `[`, `]`, or `/`. */
const TRAINING_RUN_ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

export function isValidTrainingRunId(runId: unknown): runId is string {
  return typeof runId === 'string' && TRAINING_RUN_ID_PATTERN.test(runId);
}

export function assertValidTrainingRunId(runId: unknown): asserts runId is string {
  if (!isValidTrainingRunId(runId)) {
    throw new Error(
      `Invalid training run id "${String(runId)}": must use lowercase letters, numbers, and hyphens only.`,
    );
  }
}
