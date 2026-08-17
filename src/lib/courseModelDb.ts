import * as dbApi from './dbApi';
import { pollingSubscription, type Unsubscribe } from './pollingSubscription';
import type {
  CourseModelDeploymentStatus,
  CourseModelRegistry,
  CourseModelStatus,
  CourseModelVersion,
} from '../types';
import { assertValidCourseId } from './courseId';

/**
 * Reads the per-course model registry from PostgreSQL through FastAPI.
 *
 * Course-scoped exactly like metadata, examples, and evaluations: the request
 * path is built from a validated course id and the backend keys every query on
 * it, so one course can never read another's record.
 *
 * Read-only from the application. Records are written by whoever trains and
 * promotes a model — see `scripts/register_course_model.py`. Nothing in the UI
 * creates or edits a version, because nothing in the UI trains one.
 */

const MODEL_STATUSES: readonly CourseModelStatus[] = ['ready', 'training', 'failed'];

const DEPLOYMENT_STATUSES: readonly CourseModelDeploymentStatus[] = [
  'online',
  'offline',
  'unknown',
];

function isModelStatus(value: unknown): value is CourseModelStatus {
  return typeof value === 'string' && MODEL_STATUSES.includes(value as CourseModelStatus);
}

function isDeploymentStatus(value: unknown): value is CourseModelDeploymentStatus {
  return (
    typeof value === 'string' &&
    DEPLOYMENT_STATUSES.includes(value as CourseModelDeploymentStatus)
  );
}

export function parseCourseModelVersion(value: unknown): CourseModelVersion | null {
  if (!value || typeof value !== 'object') {
    return null;
  }

  const record = value as Record<string, unknown>;

  if (
    typeof record.version !== 'string' ||
    record.version.trim() === '' ||
    typeof record.baseModel !== 'string' ||
    typeof record.artifactRef !== 'string' ||
    typeof record.createdAt !== 'string' ||
    !isModelStatus(record.status)
  ) {
    return null;
  }

  const count = Number(record.trainingExampleCount);

  return {
    version: record.version,
    baseModel: record.baseModel,
    trainingExampleCount: Number.isFinite(count) && count >= 0 ? count : 0,
    status: record.status,
    // An unrecorded deployment is "unknown", never assumed offline or online.
    deployment: isDeploymentStatus(record.deployment) ? record.deployment : 'unknown',
    artifactRef: record.artifactRef,
    createdAt: record.createdAt,
    ...(typeof record.updatedAt === 'string' ? { updatedAt: record.updatedAt } : {}),
    ...(typeof record.notes === 'string' ? { notes: record.notes } : {}),
  };
}

export function parseCourseModelRegistry(value: unknown): CourseModelRegistry | null {
  if (!value || typeof value !== 'object') {
    return null;
  }

  const record = value as Record<string, unknown>;
  const rawVersions = record.versions;

  if (!rawVersions || typeof rawVersions !== 'object') {
    return null;
  }

  const versions: Record<string, CourseModelVersion> = {};
  for (const [key, raw] of Object.entries(rawVersions as Record<string, unknown>)) {
    const parsed = parseCourseModelVersion(raw);
    if (parsed) {
      versions[key] = parsed;
    }
  }

  if (Object.keys(versions).length === 0) {
    return null;
  }

  // Fall back to the newest version rather than dropping the whole record when
  // `currentVersion` is missing or points at something unparseable.
  const currentVersion =
    typeof record.currentVersion === 'string' && versions[record.currentVersion]
      ? record.currentVersion
      : sortVersionsNewestFirst(versions)[0].version;

  return { currentVersion, versions };
}

/** Newest first, by `createdAt`, falling back to the version key. */
export function sortVersionsNewestFirst(
  versions: Record<string, CourseModelVersion>,
): CourseModelVersion[] {
  return Object.values(versions).sort((left, right) => {
    const byDate = right.createdAt.localeCompare(left.createdAt);
    return byDate !== 0 ? byDate : right.version.localeCompare(left.version);
  });
}

export function getCurrentVersion(
  registry: CourseModelRegistry,
): CourseModelVersion | null {
  return registry.versions[registry.currentVersion] ?? null;
}

/**
 * The registry for one course, or null when it has none.
 *
 * A 404 means "no model", which is a real answer and must not surface as a
 * read failure — the page says something different for each.
 */
export async function fetchCourseModel(
  courseId: string,
): Promise<CourseModelRegistry | null> {
  assertValidCourseId(courseId);

  try {
    return parseCourseModelRegistry(await dbApi.getCourseModel(courseId));
  } catch (error) {
    if (error instanceof Error && 'status' in error && error.status === 404) {
      return null;
    }
    throw error;
  }
}

export function subscribeToCourseModel(
  courseId: string,
  onData: (registry: CourseModelRegistry | null) => void,
  onError?: (message: string) => void,
): Unsubscribe {
  assertValidCourseId(courseId);

  // A registered model changes when someone trains and promotes one, which is
  // not something that happens while a professor watches this page.
  return pollingSubscription<CourseModelRegistry | null>({
    fetcher: () => fetchCourseModel(courseId),
    onData,
    onError,
  });
}
