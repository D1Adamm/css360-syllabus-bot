import * as dbApi from './dbApi';
import type { ServingSession, ServingSessionState } from '../types';

/**
 * Whether a fine-tuned serving session is up right now, read from the backend.
 *
 * This is the answer to a question the application previously could not
 * express. `CourseModelVersion.deployment` says whether one course's artifact is
 * meant to be served; it says nothing about whether a GPU is actually running
 * the service, and it cannot, because one Slurm allocation serves every course
 * whose adapter it can load.
 *
 * A session is recorded by the Tillicum start script and ends when its
 * allocation's wall clock runs out — so `null` is the normal answer most of the
 * time, and the resting state of a research GPU allocation is "nothing is
 * serving". Admin surfaces show it; professor surfaces do not.
 *
 * The compute node and port are deliberately not part of this shape. Every
 * `/api/db` route is reachable without a credential, and those two fields are
 * the only ones in the stored record that describe how to reach a machine.
 */

const SESSION_STATES: readonly ServingSessionState[] = [
  'starting',
  'ready',
  'stopped',
  'expired',
];

function isSessionState(value: unknown): value is ServingSessionState {
  return (
    typeof value === 'string' &&
    SESSION_STATES.includes(value as ServingSessionState)
  );
}

function parseCourses(value: unknown): ServingSession['courses'] {
  if (!Array.isArray(value)) {
    return undefined;
  }

  const courses = value
    .map((item) => {
      if (!item || typeof item !== 'object') {
        return null;
      }
      const record = item as Record<string, unknown>;
      if (
        typeof record.courseId !== 'string' ||
        record.courseId.trim() === '' ||
        typeof record.currentVersion !== 'string'
      ) {
        return null;
      }
      return {
        courseId: record.courseId,
        currentVersion: record.currentVersion,
      };
    })
    .filter((item): item is { courseId: string; currentVersion: string } => item !== null);

  return courses.length > 0 ? courses : undefined;
}

export function parseServingSession(value: unknown): ServingSession | null {
  if (!value || typeof value !== 'object') {
    return null;
  }

  const record = value as Record<string, unknown>;

  if (
    typeof record.sessionId !== 'string' ||
    record.sessionId.trim() === '' ||
    typeof record.jobId !== 'string' ||
    !isSessionState(record.state)
  ) {
    return null;
  }

  const courses = parseCourses(record.courses);

  return {
    sessionId: record.sessionId,
    jobId: record.jobId,
    state: record.state,
    // `live` is computed by the backend at read time from the allocation's wall
    // clock, so a session that has run out reads as expired without anything
    // having written that state. Defaulting to false rather than true: showing
    // a dead session as live is the worse mistake of the two.
    live: record.live === true,
    ...(typeof record.startedAt === 'string' ? { startedAt: record.startedAt } : {}),
    ...(typeof record.expiresAt === 'string' ? { expiresAt: record.expiresAt } : {}),
    ...(typeof record.updatedAt === 'string' ? { updatedAt: record.updatedAt } : {}),
    ...(courses ? { courses } : {}),
    ...(typeof record.baseModel === 'string' ? { baseModel: record.baseModel } : {}),
  };
}

/** The session serving right now, or null when nothing is. */
export async function fetchCurrentServingSession(): Promise<ServingSession | null> {
  const response = await dbApi.getServingSession();
  return parseServingSession(response.session);
}
