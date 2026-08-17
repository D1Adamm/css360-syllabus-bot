import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { CourseMetadata, EvaluationRecord, SeedExample } from '../types';

/**
 * Course, seed, and evaluation persistence after the PostgreSQL cutover.
 *
 * These used to assert Firebase call mechanics — which `ref` path a write went
 * to, which snapshot a listener parsed. They now assert the same guarantees one
 * layer over: that each call reaches the FastAPI route for exactly the course
 * it was given, and that what comes back is parsed the way the UI expects.
 *
 * The whole `dbApi` module is mocked, so nothing here opens a socket.
 */

const dbApiMock = vi.hoisted(() => ({
  listCourses: vi.fn(),
  getCourse: vi.fn(),
  createCourse: vi.fn(),
  updateCourse: vi.fn(),
  listSeeds: vi.fn(),
  createSeed: vi.fn(),
  updateSeed: vi.fn(),
  deleteSeed: vi.fn(),
  listEvaluations: vi.fn(),
  createEvaluation: vi.fn(),
  deleteEvaluation: vi.fn(),
  deleteAllEvaluations: vi.fn(),
}));

vi.mock('./dbApi', () => dbApiMock);

import { ApiError } from './api';
import {
  courseExists,
  createCourseMetadata,
  getCourseMetadata,
  isCourseMetadata,
  parseCourseList,
  sortCoursesNewestFirst,
  subscribeToCourses,
  updateCourseMetadata,
} from './coursesDb';
import {
  createEvaluation,
  deleteAllEvaluations,
  deleteEvaluation,
  subscribeToEvaluations,
} from './evaluationsDb';
import {
  createSeedExample,
  deleteSeedExample,
  subscribeToSeedExamples,
  updateSeedExample,
} from './seedExamplesDb';

const COURSE_A = 'css-360-winter-2026-a7rp';
const COURSE_B = 'css-350-spring-2026-n3h9';

const sampleMetadata: CourseMetadata = {
  name: 'CSS 360',
  title: 'Software Engineering',
  term: 'Spring 2026',
  instructorName: 'Instructor',
  createdAt: '2026-01-01T00:00:00.000Z',
  syllabusStatus: 'none',
  syllabusFileName: null,
  syllabusType: null,
  chunkCount: 0,
};

/** Waits for the subscription's first fetch to resolve and deliver. */
async function flush(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

beforeEach(() => {
  vi.clearAllMocks();
  dbApiMock.listCourses.mockResolvedValue({ count: 0, courses: [] });
  dbApiMock.getCourse.mockResolvedValue({
    courseId: COURSE_A,
    metadata: sampleMetadata,
  });
  dbApiMock.listSeeds.mockResolvedValue({
    courseId: COURSE_A,
    count: 0,
    seeds: [],
    reviewStatusCounts: {},
  });
  dbApiMock.listEvaluations.mockResolvedValue({
    courseId: COURSE_A,
    count: 0,
    evaluations: [],
  });
});

describe('CourseMetadata type behavior', () => {
  it('accepts valid metadata including nullable syllabus fields', () => {
    expect(isCourseMetadata(sampleMetadata)).toBe(true);
    expect(
      isCourseMetadata({
        ...sampleMetadata,
        syllabusFileName: 'syllabus.pdf',
        syllabusType: 'pdf',
      }),
    ).toBe(true);
  });

  it('rejects invalid or incomplete metadata', () => {
    expect(isCourseMetadata(null)).toBe(false);
    expect(isCourseMetadata({ ...sampleMetadata, chunkCount: 'many' })).toBe(false);
    expect(isCourseMetadata({ ...sampleMetadata, syllabusStatus: 'invented' })).toBe(
      false,
    );
  });
});

describe('course metadata through the API', () => {
  it('creates a course through the courses endpoint', async () => {
    await createCourseMetadata(COURSE_A, sampleMetadata);

    expect(dbApiMock.createCourse).toHaveBeenCalledWith(
      expect.objectContaining({ courseId: COURSE_A, name: 'CSS 360' }),
    );
  });

  it('reads metadata for the course it was given', async () => {
    const metadata = await getCourseMetadata(COURSE_A);

    expect(dbApiMock.getCourse).toHaveBeenCalledWith(COURSE_A);
    expect(metadata).toEqual(sampleMetadata);
  });

  it('reports a course that does not exist as null, not an error', async () => {
    dbApiMock.getCourse.mockRejectedValue(new ApiError('not found', 404));

    await expect(getCourseMetadata(COURSE_A)).resolves.toBeNull();
  });

  it('lets a read failure surface rather than claiming the course is gone', async () => {
    dbApiMock.getCourse.mockRejectedValue(new ApiError('Service unavailable', 503));

    await expect(getCourseMetadata(COURSE_A)).rejects.toThrow();
  });

  it('updates metadata through the course endpoint', async () => {
    await updateCourseMetadata(COURSE_A, { chunkCount: 12 });

    expect(dbApiMock.updateCourse).toHaveBeenCalledWith(COURSE_A, { chunkCount: 12 });
  });

  it('checks existence by reading the course', async () => {
    await expect(courseExists(COURSE_A)).resolves.toBe(true);

    dbApiMock.getCourse.mockRejectedValue(new ApiError('not found', 404));
    await expect(courseExists(COURSE_A)).resolves.toBe(false);
  });

  it('rejects invalid course ids before any request', async () => {
    await expect(createCourseMetadata('Bad Id', sampleMetadata)).rejects.toThrow();
    await expect(updateCourseMetadata('Bad Id', {})).rejects.toThrow();

    expect(dbApiMock.createCourse).not.toHaveBeenCalled();
    expect(dbApiMock.updateCourse).not.toHaveBeenCalled();
  });
});

describe('course list', () => {
  it('loads the course list through the API', async () => {
    dbApiMock.listCourses.mockResolvedValue({
      count: 1,
      courses: [{ courseId: COURSE_A, metadata: sampleMetadata }],
    });

    const received: unknown[] = [];
    const unsubscribe = subscribeToCourses((courses) => received.push(courses));
    await flush();
    unsubscribe();

    expect(dbApiMock.listCourses).toHaveBeenCalledTimes(1);
    expect(received[0]).toEqual([{ courseId: COURSE_A, metadata: sampleMetadata }]);
  });

  it('sorts courses newest first by createdAt', () => {
    const sorted = sortCoursesNewestFirst([
      {
        courseId: 'older',
        metadata: { ...sampleMetadata, createdAt: '2026-01-01T00:00:00.000Z' },
      },
      {
        courseId: 'newer',
        metadata: { ...sampleMetadata, createdAt: '2026-06-01T00:00:00.000Z' },
      },
    ]);

    expect(sorted.map((item) => item.courseId)).toEqual(['newer', 'older']);
  });

  it('ignores entries without usable metadata', () => {
    const items = parseCourseList([
      { courseId: COURSE_A, metadata: sampleMetadata },
      { courseId: COURSE_B, metadata: { name: 'incomplete' } as never },
    ]);

    expect(items.map((item) => item.courseId)).toEqual([COURSE_A]);
  });

  it('reports a failure through onError instead of showing an empty list', async () => {
    dbApiMock.listCourses.mockRejectedValue(new ApiError('Service unavailable', 503));

    const data: unknown[] = [];
    const errors: string[] = [];
    const unsubscribe = subscribeToCourses(
      (courses) => data.push(courses),
      (message) => errors.push(message),
    );
    await flush();
    unsubscribe();

    expect(data).toHaveLength(0);
    expect(errors[0]).toContain('Service unavailable');
  });
});

describe('seed examples through the API', () => {
  const seed: SeedExample = {
    id: 'seed-1',
    instruction: 'When are office hours?',
    response: 'Tuesdays at 2pm.',
    category: 'office hours',
    sourceSection: 'Office Hours',
    difficulty: 'Easy',
    directlyAnswered: true,
    origin: 'user',
  };

  it('creates a seed under the course it was given', async () => {
    await createSeedExample(COURSE_A, seed);

    expect(dbApiMock.createSeed).toHaveBeenCalledWith(
      COURSE_A,
      expect.objectContaining({ id: 'seed-1', instruction: 'When are office hours?' }),
    );
  });

  it('lists seeds for one course only', async () => {
    dbApiMock.listSeeds.mockResolvedValue({
      courseId: COURSE_A,
      count: 1,
      seeds: [{ ...seed, courseId: COURSE_A, question: seed.instruction }],
      reviewStatusCounts: { generated: 1 },
    });

    const received: SeedExample[][] = [];
    const unsubscribe = subscribeToSeedExamples(COURSE_A, (seeds) =>
      received.push(seeds),
    );
    await flush();
    unsubscribe();

    expect(dbApiMock.listSeeds).toHaveBeenCalledWith(COURSE_A);
    expect(received[0]?.[0]?.id).toBe('seed-1');
  });

  it('updates and deletes a seed by course and id together', async () => {
    await updateSeedExample(COURSE_A, 'seed-1', { category: 'grading' });
    await deleteSeedExample(COURSE_A, 'seed-1');

    expect(dbApiMock.updateSeed).toHaveBeenCalledWith(COURSE_A, 'seed-1', {
      category: 'grading',
    });
    expect(dbApiMock.deleteSeed).toHaveBeenCalledWith(COURSE_A, 'seed-1');
  });

  it('rejects an invalid course id before any request', async () => {
    await expect(deleteSeedExample('Bad Id', 'seed-1')).rejects.toThrow();
    expect(dbApiMock.deleteSeed).not.toHaveBeenCalled();
  });
});

describe('evaluations through the API', () => {
  const evaluation: EvaluationRecord = {
    id: 'eval-1',
    comparisonId: 'cmp-1',
    mostAccurate: 'rag',
    mostHelpful: 'rag',
    mostConcise: 'base',
    bestGrounded: 'rag',
    preferredModel: 'rag',
    hallucinationFlags: [],
    createdAt: '2026-02-01T00:00:00.000Z',
  };

  it('creates an evaluation under the course it was given', async () => {
    dbApiMock.createEvaluation.mockResolvedValue(evaluation);

    await createEvaluation(COURSE_A, evaluation);

    expect(dbApiMock.createEvaluation).toHaveBeenCalledWith(
      COURSE_A,
      expect.objectContaining({ comparisonId: 'cmp-1' }),
    );
  });

  it('lists evaluations for one course only', async () => {
    dbApiMock.listEvaluations.mockResolvedValue({
      courseId: COURSE_A,
      count: 1,
      evaluations: [evaluation],
    });

    const received: EvaluationRecord[][] = [];
    const unsubscribe = subscribeToEvaluations(COURSE_A, (items) =>
      received.push(items),
    );
    await flush();
    unsubscribe();

    expect(dbApiMock.listEvaluations).toHaveBeenCalledWith(COURSE_A);
    expect(received[0]?.[0]?.id).toBe('eval-1');
  });

  it('deletes one evaluation by course and id together', async () => {
    await deleteEvaluation(COURSE_A, 'eval-1');

    expect(dbApiMock.deleteEvaluation).toHaveBeenCalledWith(COURSE_A, 'eval-1');
  });

  it('clears a course in one request rather than one per record', async () => {
    await deleteAllEvaluations(COURSE_A, [evaluation, evaluation]);

    expect(dbApiMock.deleteAllEvaluations).toHaveBeenCalledTimes(1);
    expect(dbApiMock.deleteAllEvaluations).toHaveBeenCalledWith(COURSE_A);
  });
});
