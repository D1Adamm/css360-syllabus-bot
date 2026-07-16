import {
  onValue,
  push,
  ref,
  remove,
  set,
  update,
  type Unsubscribe,
} from 'firebase/database';
import type { SeedExample } from '../types';
import { normalizeSeedExample } from '../utils/seedDataUtils';
import { assertValidCourseId } from './courseId';
import { getCourseSeedExamplePath, getCourseSeedExamplesPath } from './coursePaths';
import { database } from './firebase';

export function getCourseSeedExamplesRef(courseId: string) {
  return ref(database, getCourseSeedExamplesPath(courseId));
}

export function getCourseSeedExampleRef(courseId: string, exampleId: string) {
  return ref(database, getCourseSeedExamplePath(courseId, exampleId));
}

export function parseSeedExamplesFromSnapshot(data: unknown): SeedExample[] {
  if (!data || typeof data !== 'object') {
    return [];
  }

  const seeds: SeedExample[] = [];

  for (const [key, value] of Object.entries(data as Record<string, unknown>)) {
    const normalized = normalizeSeedExample(value, key);
    if (normalized) {
      seeds.push(normalized);
    }
  }

  return seeds.sort((left, right) => {
    const leftTime = left.createdAt ?? '';
    const rightTime = right.createdAt ?? '';
    return rightTime.localeCompare(leftTime);
  });
}

/** Subscribe to courses/{courseId}/seedExamples. */
export function subscribeToSeedExamples(
  courseId: string,
  onData: (seeds: SeedExample[]) => void,
  onError: (message: string) => void,
): Unsubscribe {
  assertValidCourseId(courseId);
  const seedsRef = getCourseSeedExamplesRef(courseId);

  return onValue(
    seedsRef,
    (snapshot) => {
      onData(parseSeedExamplesFromSnapshot(snapshot.val()));
    },
    (error) => {
      onError(error.message);
    },
  );
}

/** Create a seed example under courses/{courseId}/seedExamples. */
export async function createSeedExample(
  courseId: string,
  seed: SeedExample,
): Promise<void> {
  assertValidCourseId(courseId);
  const seedRef = push(getCourseSeedExamplesRef(courseId));
  const storedSeed: SeedExample = {
    ...seed,
    id: seedRef.key ?? seed.id,
    createdAt: seed.createdAt ?? new Date().toISOString(),
  };

  await set(seedRef, storedSeed);
}

export async function updateSeedExample(
  courseId: string,
  exampleId: string,
  updates: Partial<SeedExample>,
): Promise<void> {
  assertValidCourseId(courseId);
  await update(getCourseSeedExampleRef(courseId, exampleId), updates);
}

/** Delete a seed example from courses/{courseId}/seedExamples/{exampleId}. */
export async function deleteSeedExample(
  courseId: string,
  exampleId: string,
): Promise<void> {
  assertValidCourseId(courseId);
  await remove(getCourseSeedExampleRef(courseId, exampleId));
}

export async function deleteAllSeedExamples(
  courseId: string,
  seeds: SeedExample[],
): Promise<void> {
  await Promise.all(seeds.map((seed) => deleteSeedExample(courseId, seed.id)));
}

/** Delete only student-created seeds (origin "user"). */
export async function deleteAllUserSeedExamples(
  courseId: string,
  seeds: SeedExample[],
): Promise<void> {
  const userSeeds = seeds.filter((seed) => seed.origin === 'user');
  await deleteAllSeedExamples(courseId, userSeeds);
}
