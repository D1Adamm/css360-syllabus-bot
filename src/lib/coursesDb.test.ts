import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { CourseMetadata, EvaluationRecord, SeedExample } from '../types';

const { setMock, getMock, updateMock, removeMock, pushMock, onValueMock, refMock } =
  vi.hoisted(() => ({
    setMock: vi.fn(),
    getMock: vi.fn(),
    updateMock: vi.fn(),
    removeMock: vi.fn(),
    pushMock: vi.fn(),
    onValueMock: vi.fn(),
    refMock: vi.fn((_db: unknown, path: string) => ({ path })),
  }));

vi.mock('./firebase', () => ({
  database: { name: 'mock-db' },
}));

vi.mock('firebase/database', () => ({
  ref: refMock,
  set: setMock,
  get: getMock,
  update: updateMock,
  remove: removeMock,
  push: pushMock,
  onValue: onValueMock,
}));

import {
  courseExists,
  createCourseMetadata,
  getCourseMetadata,
  isCourseMetadata,
  updateCourseMetadata,
} from './coursesDb';
import {
  createEvaluation,
  deleteEvaluation,
  subscribeToEvaluations,
} from './evaluationsDb';
import {
  createSeedExample,
  deleteSeedExample,
  subscribeToSeedExamples,
  updateSeedExample,
} from './seedExamplesDb';

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

const sampleSeed: SeedExample = {
  id: 'seed-local',
  instruction: 'When does class meet?',
  response: 'Tuesday and Thursday.',
  category: 'Course Basics',
  sourceSection: 'Course Meetings',
  difficulty: 'Easy',
  directlyAnswered: true,
  origin: 'user',
};

const sampleEvaluation: EvaluationRecord = {
  id: 'eval-local',
  comparisonId: 'comparison-001',
  mostAccurate: 'rag',
  mostHelpful: 'rag',
  mostConcise: 'base',
  bestGrounded: 'rag',
  preferredModel: 'rag',
  hallucinationFlags: ['base'],
  createdAt: '2026-01-02T00:00:00.000Z',
};

describe('CourseMetadata type behavior', () => {
  it('accepts valid metadata including nullable syllabus fields', () => {
    expect(isCourseMetadata(sampleMetadata)).toBe(true);
    expect(
      isCourseMetadata({
        ...sampleMetadata,
        syllabusStatus: 'ready',
        syllabusFileName: 'syllabus.pdf',
        syllabusType: 'application/pdf',
        chunkCount: 12,
      }),
    ).toBe(true);
  });

  it('rejects invalid or incomplete metadata', () => {
    expect(isCourseMetadata(null)).toBe(false);
    expect(isCourseMetadata({ ...sampleMetadata, syllabusStatus: 'unknown' })).toBe(
      false,
    );
    expect(isCourseMetadata({ ...sampleMetadata, chunkCount: '0' })).toBe(false);
    expect(isCourseMetadata({ ...sampleMetadata, syllabusFileName: 1 })).toBe(false);
  });
});

describe('course metadata Firebase helpers', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    pushMock.mockReturnValue({ key: 'generated-id' });
    setMock.mockResolvedValue(undefined);
    updateMock.mockResolvedValue(undefined);
    removeMock.mockResolvedValue(undefined);
  });

  it('writes metadata to courses/{courseId}/metadata', async () => {
    await createCourseMetadata('css360-default', sampleMetadata);

    expect(refMock).toHaveBeenCalledWith(
      expect.anything(),
      'courses/css360-default/metadata',
    );
    expect(setMock).toHaveBeenCalledWith(
      { path: 'courses/css360-default/metadata' },
      sampleMetadata,
    );
  });

  it('reads metadata from courses/{courseId}/metadata', async () => {
    getMock.mockResolvedValue({
      exists: () => true,
      val: () => sampleMetadata,
    });

    const result = await getCourseMetadata('css360-default');

    expect(refMock).toHaveBeenCalledWith(
      expect.anything(),
      'courses/css360-default/metadata',
    );
    expect(result).toEqual(sampleMetadata);
  });

  it('updates metadata at courses/{courseId}/metadata', async () => {
    await updateCourseMetadata('css360-default', { chunkCount: 5, syllabusStatus: 'ready' });

    expect(refMock).toHaveBeenCalledWith(
      expect.anything(),
      'courses/css360-default/metadata',
    );
    expect(updateMock).toHaveBeenCalledWith(
      { path: 'courses/css360-default/metadata' },
      { chunkCount: 5, syllabusStatus: 'ready' },
    );
  });

  it('checks course existence via metadata path', async () => {
    getMock.mockResolvedValue({ exists: () => true });
    await expect(courseExists('css360-default')).resolves.toBe(true);

    getMock.mockResolvedValue({ exists: () => false });
    await expect(courseExists('css360-default')).resolves.toBe(false);
  });

  it('rejects invalid course ids before writing metadata', async () => {
    await expect(createCourseMetadata('../evil', sampleMetadata)).rejects.toThrow(
      /Invalid courseId/,
    );
    expect(setMock).not.toHaveBeenCalled();
  });
});

describe('course-aware seed-example helpers', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    pushMock.mockReturnValue({ key: 'generated-seed' });
    setMock.mockResolvedValue(undefined);
    updateMock.mockResolvedValue(undefined);
    removeMock.mockResolvedValue(undefined);
    onValueMock.mockReturnValue(() => undefined);
  });

  it('creates seed examples under courses/{courseId}/seedExamples', async () => {
    await createSeedExample('css360-default', sampleSeed);

    expect(refMock).toHaveBeenCalledWith(
      expect.anything(),
      'courses/css360-default/seedExamples',
    );
    expect(setMock).toHaveBeenCalledWith(
      { key: 'generated-seed' },
      expect.objectContaining({
        id: 'generated-seed',
        instruction: sampleSeed.instruction,
      }),
    );
  });

  it('subscribes to courses/{courseId}/seedExamples', () => {
    subscribeToSeedExamples(
      'css360-default',
      () => undefined,
      () => undefined,
    );

    expect(refMock).toHaveBeenCalledWith(
      expect.anything(),
      'courses/css360-default/seedExamples',
    );
    expect(onValueMock).toHaveBeenCalled();
  });

  it('updates and deletes seed examples under the course path', async () => {
    await updateSeedExample('css360-default', 'seed-1', { notes: 'reviewed' });
    expect(refMock).toHaveBeenCalledWith(
      expect.anything(),
      'courses/css360-default/seedExamples/seed-1',
    );
    expect(updateMock).toHaveBeenCalledWith(
      { path: 'courses/css360-default/seedExamples/seed-1' },
      { notes: 'reviewed' },
    );

    await deleteSeedExample('css360-default', 'seed-1');
    expect(removeMock).toHaveBeenCalledWith({
      path: 'courses/css360-default/seedExamples/seed-1',
    });
  });

  it('preserves legacy global seed path when courseId is omitted', async () => {
    await createSeedExample(sampleSeed);

    expect(refMock).toHaveBeenCalledWith(expect.anything(), 'seedExamples');
  });
});

describe('course-aware evaluation helpers', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    pushMock.mockReturnValue({ key: 'generated-eval' });
    setMock.mockResolvedValue(undefined);
    removeMock.mockResolvedValue(undefined);
    onValueMock.mockReturnValue(() => undefined);
  });

  it('creates evaluations under courses/{courseId}/evaluations', async () => {
    const stored = await createEvaluation('css360-default', sampleEvaluation);

    expect(refMock).toHaveBeenCalledWith(
      expect.anything(),
      'courses/css360-default/evaluations',
    );
    expect(stored.id).toBe('generated-eval');
    expect(setMock).toHaveBeenCalledWith(
      { key: 'generated-eval' },
      expect.objectContaining({
        id: 'generated-eval',
        comparisonId: sampleEvaluation.comparisonId,
      }),
    );
  });

  it('subscribes to courses/{courseId}/evaluations', () => {
    subscribeToEvaluations(
      'css360-default',
      () => undefined,
      () => undefined,
    );

    expect(refMock).toHaveBeenCalledWith(
      expect.anything(),
      'courses/css360-default/evaluations',
    );
    expect(onValueMock).toHaveBeenCalled();
  });

  it('deletes evaluations under the course path', async () => {
    await deleteEvaluation('css360-default', 'eval-1');

    expect(refMock).toHaveBeenCalledWith(
      expect.anything(),
      'courses/css360-default/evaluations/eval-1',
    );
    expect(removeMock).toHaveBeenCalledWith({
      path: 'courses/css360-default/evaluations/eval-1',
    });
  });

  it('preserves legacy global evaluation path when courseId is omitted', async () => {
    await createEvaluation(sampleEvaluation);

    expect(refMock).toHaveBeenCalledWith(expect.anything(), 'evaluations');
  });
});
