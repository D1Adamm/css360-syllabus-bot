import { assertValidCourseId } from './courseId';

export function getCourseMetadataPath(courseId: string): string {
  assertValidCourseId(courseId);
  return `courses/${courseId}/metadata`;
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
