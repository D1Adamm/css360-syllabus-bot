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
import { isSeedExample } from '../utils/seedDataUtils';
import { assertValidCourseId } from './courseId';
import { getCourseSeedExamplePath, getCourseSeedExamplesPath } from './coursePaths';
import { database } from './firebase';

/** Legacy global path used by the current UI. Not yet migrated under courses/. */
export const SEED_EXAMPLES_PATH = 'seedExamples';

export function getSeedExamplesRef() {
  return ref(database, SEED_EXAMPLES_PATH);
}

export function getSeedExampleRef(id: string) {
  return ref(database, `${SEED_EXAMPLES_PATH}/${id}`);
}

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

  return Object.values(data)
    .filter(isSeedExample)
    .sort((left, right) => {
      const leftTime = left.createdAt ?? '';
      const rightTime = right.createdAt ?? '';
      return rightTime.localeCompare(leftTime);
    });
}

/**
 * Subscribe to seed examples.
 * - Global (legacy UI): subscribeToSeedExamples(onData, onError)
 * - Course-aware: subscribeToSeedExamples(courseId, onData, onError)
 */
export function subscribeToSeedExamples(
  onData: (seeds: SeedExample[]) => void,
  onError: (message: string) => void,
): Unsubscribe;
export function subscribeToSeedExamples(
  courseId: string,
  onData: (seeds: SeedExample[]) => void,
  onError: (message: string) => void,
): Unsubscribe;
export function subscribeToSeedExamples(
  courseIdOrOnData: string | ((seeds: SeedExample[]) => void),
  onDataOrOnError: ((seeds: SeedExample[]) => void) | ((message: string) => void),
  maybeOnError?: (message: string) => void,
): Unsubscribe {
  if (typeof courseIdOrOnData === 'string') {
    assertValidCourseId(courseIdOrOnData);
    const onData = onDataOrOnError as (seeds: SeedExample[]) => void;
    const onError = maybeOnError as (message: string) => void;
    const seedsRef = getCourseSeedExamplesRef(courseIdOrOnData);

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

  const onData = courseIdOrOnData;
  const onError = onDataOrOnError as (message: string) => void;
  const seedsRef = getSeedExamplesRef();

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

/**
 * Create a seed example.
 * - Global (legacy UI): createSeedExample(seedExample)
 * - Course-aware: createSeedExample(courseId, seedExample)
 */
export async function createSeedExample(seed: SeedExample): Promise<void>;
export async function createSeedExample(courseId: string, seed: SeedExample): Promise<void>;
export async function createSeedExample(
  courseIdOrSeed: string | SeedExample,
  maybeSeed?: SeedExample,
): Promise<void> {
  if (typeof courseIdOrSeed === 'string') {
    assertValidCourseId(courseIdOrSeed);
    const seed = maybeSeed as SeedExample;
    const seedRef = push(getCourseSeedExamplesRef(courseIdOrSeed));
    const storedSeed: SeedExample = {
      ...seed,
      id: seedRef.key ?? seed.id,
      createdAt: seed.createdAt ?? new Date().toISOString(),
    };

    await set(seedRef, storedSeed);
    return;
  }

  const seed = courseIdOrSeed;
  const seedRef = push(getSeedExamplesRef());
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

/**
 * Delete a seed example.
 * - Global (legacy UI): deleteSeedExample(exampleId)
 * - Course-aware: deleteSeedExample(courseId, exampleId)
 */
export async function deleteSeedExample(id: string): Promise<void>;
export async function deleteSeedExample(courseId: string, exampleId: string): Promise<void>;
export async function deleteSeedExample(
  courseIdOrId: string,
  maybeExampleId?: string,
): Promise<void> {
  if (maybeExampleId !== undefined) {
    assertValidCourseId(courseIdOrId);
    await remove(getCourseSeedExampleRef(courseIdOrId, maybeExampleId));
    return;
  }

  await remove(getSeedExampleRef(courseIdOrId));
}

export async function deleteAllSeedExamples(
  courseId: string,
  seeds: SeedExample[],
): Promise<void> {
  await Promise.all(seeds.map((seed) => deleteSeedExample(courseId, seed.id)));
}
