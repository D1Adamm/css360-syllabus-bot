import { describe, expect, it, vi } from 'vitest';

vi.mock('./firebase', () => ({ database: {}, app: {} }));

import {
  getCurrentVersion,
  parseCourseModelRegistry,
  parseCourseModelVersion,
  sortVersionsNewestFirst,
} from './courseModelDb';

const CSS360_V1 = {
  version: 'v1',
  baseModel: 'meta-llama/Llama-3.2-3B-Instruct',
  trainingExampleCount: 54,
  status: 'ready',
  deployment: 'offline',
  artifactRef: 'css-360-qlora/adapter',
  createdAt: '2026-08-11T06:22:50.979Z',
};

describe('parseCourseModelVersion', () => {
  it('reads the CSS 360 record as stored', () => {
    const parsed = parseCourseModelVersion(CSS360_V1);

    expect(parsed).toMatchObject({
      version: 'v1',
      baseModel: 'meta-llama/Llama-3.2-3B-Instruct',
      trainingExampleCount: 54,
      status: 'ready',
      // Trained and saved, but nothing is serving it. Both facts, separately.
      deployment: 'offline',
      artifactRef: 'css-360-qlora/adapter',
    });
  });

  it('treats an unrecorded deployment as unknown, never as offline', () => {
    const { deployment, ...withoutDeployment } = CSS360_V1;
    void deployment;

    expect(parseCourseModelVersion(withoutDeployment)?.deployment).toBe('unknown');
  });

  it('rejects a record missing the fields that make it meaningful', () => {
    expect(parseCourseModelVersion(null)).toBeNull();
    expect(parseCourseModelVersion({})).toBeNull();
    expect(parseCourseModelVersion({ ...CSS360_V1, status: 'deployed' })).toBeNull();
    expect(parseCourseModelVersion({ ...CSS360_V1, version: '' })).toBeNull();
    expect(parseCourseModelVersion({ ...CSS360_V1, artifactRef: 42 })).toBeNull();
  });

  it('coerces a nonsense training count rather than dropping the record', () => {
    const parsed = parseCourseModelVersion({
      ...CSS360_V1,
      trainingExampleCount: 'many',
    });
    expect(parsed?.trainingExampleCount).toBe(0);
    expect(parsed?.status).toBe('ready');
  });
});

describe('parseCourseModelRegistry', () => {
  it('parses a single-version registry', () => {
    const registry = parseCourseModelRegistry({
      currentVersion: 'v1',
      versions: { v1: CSS360_V1 },
    });

    expect(registry?.currentVersion).toBe('v1');
    expect(getCurrentVersion(registry!)?.trainingExampleCount).toBe(54);
  });

  it('returns null when a course has no model', () => {
    expect(parseCourseModelRegistry(null)).toBeNull();
    expect(parseCourseModelRegistry({})).toBeNull();
    expect(parseCourseModelRegistry({ versions: {} })).toBeNull();
  });

  it('falls back to the newest version when currentVersion is unusable', () => {
    const registry = parseCourseModelRegistry({
      currentVersion: 'v9',
      versions: {
        v1: CSS360_V1,
        v2: { ...CSS360_V1, version: 'v2', createdAt: '2026-09-01T00:00:00.000Z' },
      },
    });

    expect(registry?.currentVersion).toBe('v2');
  });

  it('keeps the good versions when one is malformed', () => {
    const registry = parseCourseModelRegistry({
      currentVersion: 'v1',
      versions: { v1: CSS360_V1, v2: { version: 'v2' } },
    });

    expect(Object.keys(registry!.versions)).toEqual(['v1']);
  });
});

describe('sortVersionsNewestFirst', () => {
  it('orders history newest first', () => {
    const versions = {
      v1: CSS360_V1,
      v2: { ...CSS360_V1, version: 'v2', createdAt: '2026-09-01T00:00:00.000Z' },
      v3: { ...CSS360_V1, version: 'v3', createdAt: '2026-10-01T00:00:00.000Z' },
    };
    const registry = parseCourseModelRegistry({ currentVersion: 'v2', versions });

    expect(sortVersionsNewestFirst(registry!.versions).map((v) => v.version)).toEqual([
      'v3',
      'v2',
      'v1',
    ]);
    // Newest is not automatically current — promotion is explicit.
    expect(registry?.currentVersion).toBe('v2');
  });
});
