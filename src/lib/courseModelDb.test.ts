import { describe, expect, it } from 'vitest';
import type { CourseModelRegistry, CourseModelVersion } from '../types';

import {
  getCurrentVersion,
  getDisplayVersion,
  getNewestVersion,
  getServingVersion,
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

/*
 * Which version a professor is told about (audit finding B4, 2026-11-03).
 *
 * `currentVersion` is set to the newest registered version by the run that
 * registers it. The version students actually get is the activated one —
 * `deployment = online`, `status = ready` — which after a retrain is older.
 */
describe('getServingVersion / getNewestVersion / getDisplayVersion', () => {
  function version(
    name: string,
    status: CourseModelVersion['status'],
    deployment: CourseModelVersion['deployment'],
    createdAt: string,
  ): CourseModelVersion {
    return {
      version: name,
      baseModel: 'meta-llama/Llama-3.2-3B-Instruct',
      trainingExampleCount: 37,
      status,
      deployment,
      artifactRef: `serving/css-360/${name}/adapter`,
      createdAt,
    };
  }

  const v1 = version('v1', 'ready', 'online', '2026-08-01T00:00:00.000Z');
  const v2 = version('v2', 'ready', 'offline', '2026-09-01T00:00:00.000Z');
  const v3training = version('v3', 'training', 'unknown', '2026-10-01T00:00:00.000Z');

  it('serves the activated version, not the newest registered one', () => {
    const registry: CourseModelRegistry = { currentVersion: 'v2', versions: { v1, v2 } };
    expect(getServingVersion(registry)?.version).toBe('v1');
    expect(getNewestVersion(registry)?.version).toBe('v2');
    expect(getDisplayVersion(registry)?.version).toBe('v1');
  });

  it('serves nothing when no version is activated, and then describes the newest', () => {
    const registry: CourseModelRegistry = {
      currentVersion: 'v3',
      versions: { v2, v3: v3training },
    };
    expect(getServingVersion(registry)).toBeNull();
    expect(getDisplayVersion(registry)?.version).toBe('v3');
  });

  it('does not count an online row that is not ready as serving', () => {
    const registry: CourseModelRegistry = {
      currentVersion: 'v2',
      versions: {
        v1: { ...v1, deployment: 'offline' },
        v2: { ...v2, deployment: 'online', status: 'failed' },
      },
    };
    expect(getServingVersion(registry)).toBeNull();
    // Nothing activated: the newest registered version is what is described.
    expect(getDisplayVersion(registry)?.version).toBe('v2');
  });

  it('takes the highest activated version when more than one row claims it, v10 after v9', () => {
    const registry: CourseModelRegistry = {
      currentVersion: 'v10',
      versions: {
        v9: version('v9', 'ready', 'online', '2026-10-09T00:00:00.000Z'),
        v10: version('v10', 'ready', 'online', '2026-10-10T00:00:00.000Z'),
        v2: version('v2', 'ready', 'online', '2026-10-02T00:00:00.000Z'),
      },
    };
    expect(getServingVersion(registry)?.version).toBe('v10');
  });
});
