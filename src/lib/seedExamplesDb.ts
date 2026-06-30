import { onValue, push, ref, remove, set, type Unsubscribe } from 'firebase/database';
import type { SeedExample } from '../types';
import { isSeedExample } from '../utils/seedDataUtils';
import { database } from './firebase';

export const SEED_EXAMPLES_PATH = 'seedExamples';

export function getSeedExamplesRef() {
  return ref(database, SEED_EXAMPLES_PATH);
}

export function getSeedExampleRef(id: string) {
  return ref(database, `${SEED_EXAMPLES_PATH}/${id}`);
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

export function subscribeToSeedExamples(
  onData: (seeds: SeedExample[]) => void,
  onError: (message: string) => void,
): Unsubscribe {
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

export async function createSeedExample(seed: SeedExample): Promise<void> {
  const seedRef = push(getSeedExamplesRef());
  const storedSeed: SeedExample = {
    ...seed,
    id: seedRef.key ?? seed.id,
    createdAt: seed.createdAt ?? new Date().toISOString(),
  };

  await set(seedRef, storedSeed);
}

export async function deleteSeedExample(id: string): Promise<void> {
  await remove(getSeedExampleRef(id));
}

export async function deleteAllSeedExamples(seeds: SeedExample[]): Promise<void> {
  await Promise.all(seeds.map((seed) => deleteSeedExample(seed.id)));
}
