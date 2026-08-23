import type { SeedExample } from '../types';
import { normalizeSeedExample } from '../utils/seedDataUtils';
import { assertValidCourseId } from './courseId';
import * as dbApi from './dbApi';
import { pollingSubscription, type Unsubscribe } from './pollingSubscription';

/**
 * Seed examples, now read and written through FastAPI against PostgreSQL.
 *
 * `normalizeSeedExample` still runs over every record. The API returns both
 * name pairs (instruction/question, response/answer) as the stored records do,
 * and that normalizer is what the seed table, the review queue, and the export
 * screens all agree on — running it here means the cutover changed the source
 * of the records and nothing about their shape.
 */

export function parseSeedExampleList(seeds: dbApi.DbSeedRecord[]): SeedExample[] {
  const normalized: SeedExample[] = [];

  for (const seed of seeds) {
    const parsed = normalizeSeedExample(seed, seed?.id);
    if (parsed) {
      normalized.push(parsed);
    }
  }

  // The API already orders newest first; sorting again keeps this independent
  // of that promise, exactly as the pre-cutover version did.
  return normalized.sort((left, right) => {
    const leftTime = left.createdAt ?? '';
    const rightTime = right.createdAt ?? '';
    return rightTime.localeCompare(leftTime);
  });
}

export async function fetchSeedExamples(courseId: string): Promise<SeedExample[]> {
  assertValidCourseId(courseId);
  return parseSeedExampleList((await dbApi.listSeeds(courseId)).seeds);
}

export function subscribeToSeedExamples(
  courseId: string,
  onData: (seeds: SeedExample[]) => void,
  onError?: (message: string) => void,
): Unsubscribe {
  assertValidCourseId(courseId);

  return pollingSubscription<SeedExample[]>({
    fetcher: () => fetchSeedExamples(courseId),
    onData,
    onError,
  });
}

export async function createSeedExample(
  courseId: string,
  seed: SeedExample,
): Promise<void> {
  assertValidCourseId(courseId);

  await dbApi.createSeed(courseId, {
    ...seed,
    createdAt: seed.createdAt ?? new Date().toISOString(),
  });
}

export async function updateSeedExample(
  courseId: string,
  exampleId: string,
  updates: Partial<SeedExample>,
): Promise<void> {
  assertValidCourseId(courseId);
  await dbApi.updateSeed(courseId, exampleId, updates);
}

export async function deleteSeedExample(
  courseId: string,
  exampleId: string,
): Promise<void> {
  assertValidCourseId(courseId);
  await dbApi.deleteSeed(courseId, exampleId);
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
