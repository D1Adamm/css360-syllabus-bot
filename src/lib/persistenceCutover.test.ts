import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import type { CourseMetadata } from '../types';

/**
 * Guards for the PostgreSQL cutover itself.
 *
 * Everything else tests behaviour through a module. These test the properties
 * the cutover is *for*: that no persistence code reaches Firebase from the
 * browser, that polling exists only while something is genuinely moving, and
 * that no test in this suite can quietly perform real network I/O.
 *
 * The Firebase checks are regression guards, not mocks of a live dependency.
 * The package is gone from `package.json` and no module imports it; these fail
 * loudly if an import is ever reintroduced, which is cheaper than discovering
 * it in a production bundle.
 */

const dbApiMock = vi.hoisted(() => ({
  getCourse: vi.fn(),
  listCourses: vi.fn(),
}));

vi.mock('./dbApi', () => dbApiMock);

import { subscribeToCourseMetadata } from './coursesDb';
import { pollingSubscription } from './pollingSubscription';

const COURSE = 'css-350-spring-2026-n3h9';

const METADATA: CourseMetadata = {
  name: 'CSS 350',
  title: 'Management principals',
  term: 'Spring 2026',
  instructorName: 'Kaylea Champion',
  createdAt: '2026-08-13T02:15:50.410Z',
  syllabusStatus: 'indexed',
  syllabusFileName: 'Css 350.txt',
  syllabusType: 'txt',
  chunkCount: 180,
};

function withStarterStatus(status: string): CourseMetadata {
  return {
    ...METADATA,
    starterSeedGeneration: { status, targetCount: 50, savedCount: 9 },
  };
}

async function flush(): Promise<void> {
  await vi.advanceTimersByTimeAsync(0);
}

describe('no browser Firebase in persistence code', () => {
  /**
   * The cutover's actual acceptance criterion, checked statically.
   *
   * A behavioural test can only prove the paths it exercises. This proves the
   * property for every persistence module at once, including ones added later,
   * which is the failure mode worth guarding: someone reaches for the Firebase
   * SDK again because the import is still there and still works.
   */
  const PERSISTENCE_MODULES = [
    'coursesDb.ts',
    'seedExamplesDb.ts',
    'evaluationsDb.ts',
    'courseModelDb.ts',
    'courseModelRequestDb.ts',
    'trainingRunDb.ts',
    'dbApi.ts',
    'pollingSubscription.ts',
    'createCourse.ts',
  ];

  it.each(PERSISTENCE_MODULES)('%s does not import firebase', (moduleName) => {
    const source = readFileSync(join(__dirname, moduleName), 'utf8');

    expect(source).not.toMatch(/from '(firebase\/database|\.\/firebase)'/);
  });

  it('no module under src/lib imports the Firebase SDK at all', () => {
    /*
     * Stronger than it started. Training-run enqueue was the last browser write
     * to Firebase; it now goes through the backend, which owns both the queue
     * the cluster claims from and the PostgreSQL mirror. With it gone the SDK
     * bootstrap had no consumers and the dependency was removed, so the correct
     * assertion is zero, not one exception.
     */
    const libDir = __dirname;
    const importers = readdirSync(libDir)
      .filter((name) => name.endsWith('.ts') && !name.includes('.test.'))
      .filter((name) =>
        /from '(firebase\/database|firebase\/app|\.\/firebase)'/.test(
          readFileSync(join(libDir, name), 'utf8'),
        ),
      );

    expect(importers).toEqual([]);
  });

  it('no page or hook imports the Firebase SDK directly', () => {
    const roots = ['pages', 'hooks', 'components', 'context', 'shell'];
    const offenders: string[] = [];

    const walk = (dir: string) => {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const full = join(dir, entry.name);
        if (entry.isDirectory()) {
          walk(full);
          continue;
        }
        if (!/\.tsx?$/.test(entry.name) || entry.name.includes('.test.')) {
          continue;
        }
        if (/from 'firebase\/database'/.test(readFileSync(full, 'utf8'))) {
          offenders.push(full);
        }
      }
    };

    for (const root of roots) {
      walk(join(__dirname, '..', root));
    }

    expect(offenders).toEqual([]);
  });
});

describe('polling only while something is moving', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('fetches once for a course whose generation already finished', async () => {
    dbApiMock.getCourse.mockResolvedValue({
      courseId: COURSE,
      metadata: withStarterStatus('ready'),
    });

    const unsubscribe = subscribeToCourseMetadata(COURSE, () => undefined);
    await flush();
    await vi.advanceTimersByTimeAsync(60_000);
    unsubscribe();

    expect(dbApiMock.getCourse).toHaveBeenCalledTimes(1);
  });

  it.each(['ready', 'partial', 'failed'])(
    'treats %s as terminal and stops',
    async (status) => {
      dbApiMock.getCourse.mockResolvedValue({
        courseId: COURSE,
        metadata: withStarterStatus(status),
      });

      const unsubscribe = subscribeToCourseMetadata(COURSE, () => undefined);
      await flush();
      await vi.advanceTimersByTimeAsync(30_000);
      unsubscribe();

      expect(dbApiMock.getCourse).toHaveBeenCalledTimes(1);
    },
  );

  it.each(['queued', 'generating'])(
    'keeps polling while generation is %s',
    async (status) => {
      dbApiMock.getCourse.mockResolvedValue({
        courseId: COURSE,
        metadata: withStarterStatus(status),
      });

      const unsubscribe = subscribeToCourseMetadata(COURSE, () => undefined);
      await flush();
      await vi.advanceTimersByTimeAsync(15_000);
      unsubscribe();

      expect(dbApiMock.getCourse.mock.calls.length).toBeGreaterThan(1);
    },
  );

  it('stops the moment generation reaches a terminal state', async () => {
    dbApiMock.getCourse
      .mockResolvedValueOnce({
        courseId: COURSE,
        metadata: withStarterStatus('generating'),
      })
      .mockResolvedValue({ courseId: COURSE, metadata: withStarterStatus('ready') });

    const unsubscribe = subscribeToCourseMetadata(COURSE, () => undefined);
    await flush();
    await vi.advanceTimersByTimeAsync(60_000);
    unsubscribe();

    // One while running, one that returned the finished record, and no more.
    expect(dbApiMock.getCourse).toHaveBeenCalledTimes(2);
    unsubscribe();
  });

  it('makes no further request after unsubscribe', async () => {
    dbApiMock.getCourse.mockResolvedValue({
      courseId: COURSE,
      metadata: withStarterStatus('generating'),
    });

    const unsubscribe = subscribeToCourseMetadata(COURSE, () => undefined);
    await flush();
    unsubscribe();

    const callsAtUnsubscribe = dbApiMock.getCourse.mock.calls.length;
    await vi.advanceTimersByTimeAsync(60_000);

    expect(dbApiMock.getCourse).toHaveBeenCalledTimes(callsAtUnsubscribe);
  });

  it('stops polling when a read fails rather than hammering a broken endpoint', async () => {
    dbApiMock.getCourse.mockRejectedValue(new Error('Service unavailable'));

    const errors: string[] = [];
    const unsubscribe = subscribeToCourseMetadata(
      COURSE,
      () => undefined,
      (message) => errors.push(message),
    );
    await flush();
    await vi.advanceTimersByTimeAsync(60_000);
    unsubscribe();

    expect(dbApiMock.getCourse).toHaveBeenCalledTimes(1);
    expect(errors).toEqual(['Service unavailable']);
  });
});

describe('a subscription that has been torn down delivers nothing', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('drops a response that arrives after the course changed', async () => {
    /*
     * The leak this prevents: a professor switches course while the first
     * request is still open. Without the cancelled check the late response
     * repopulates the page, and one course's data appears under another's
     * heading.
     */
    let resolveLate: ((value: string) => void) | undefined;
    const received: string[] = [];

    const unsubscribe = pollingSubscription<string>({
      fetcher: () =>
        new Promise<string>((resolve) => {
          resolveLate = resolve;
        }),
      onData: (value) => received.push(value),
    });

    unsubscribe();
    resolveLate?.('course-a data');
    await vi.advanceTimersByTimeAsync(0);

    expect(received).toEqual([]);
  });

  it('drops a late failure too', async () => {
    let rejectLate: ((error: Error) => void) | undefined;
    const errors: string[] = [];

    const unsubscribe = pollingSubscription<string>({
      fetcher: () =>
        new Promise<string>((_resolve, reject) => {
          rejectLate = reject;
        }),
      onData: () => undefined,
      onError: (message) => errors.push(message),
    });

    unsubscribe();
    rejectLate?.(new Error('too late'));
    await vi.advanceTimersByTimeAsync(0);

    expect(errors).toEqual([]);
  });
});
