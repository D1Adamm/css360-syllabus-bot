import { beforeEach, describe, expect, it, vi } from 'vitest';

const listCourseSeeds = vi.fn();
const exportApprovedCourseSeeds = vi.fn();
const prepareTrainingSplit = vi.fn();
const updateCourseModelRequest = vi.fn();

vi.mock('./api', () => ({
  listCourseSeeds: (...args: unknown[]) => listCourseSeeds(...args),
  exportApprovedCourseSeeds: (...args: unknown[]) => exportApprovedCourseSeeds(...args),
  prepareTrainingSplit: (...args: unknown[]) => prepareTrainingSplit(...args),
}));

vi.mock('./courseModelRequestDb', () => ({
  updateCourseModelRequest: (...args: unknown[]) => updateCourseModelRequest(...args),
}));

import {
  datasetRefForCourse,
  InsufficientApprovedExamplesError,
  prepareTrainingDataForRequest,
} from './prepareTrainingData';

const COURSE_490 = 'css-490-spring-2026-cgvl';
const COURSE_360 = 'css-360-winter-2026-a7rp';

function approvedSeeds(count: number, prefix = 'c490') {
  return Array.from({ length: count }, (_, index) => ({
    id: `${prefix}-${index}`,
    question: `${prefix} question ${index}`,
    answer: `${prefix} answer ${index}`,
    reviewStatus: 'approved',
  }));
}

beforeEach(() => {
  vi.clearAllMocks();

  listCourseSeeds.mockResolvedValue({ seeds: approvedSeeds(42) });
  exportApprovedCourseSeeds.mockResolvedValue({
    courseId: COURSE_490,
    summary: { approvedCount: 42, validatedCount: 42 },
  });
  prepareTrainingSplit.mockResolvedValue({
    courseId: COURSE_490,
    summary: { trainExamples: 38, validationExamples: 4, totalExamples: 42, splitSeed: 360 },
  });
  updateCourseModelRequest.mockResolvedValue(undefined);
});

describe('successful preparation', () => {
  it('exports and splits using the existing endpoints, in order', async () => {
    await prepareTrainingDataForRequest(COURSE_490);

    // Re-count first, then export, then split. Splitting before exporting would
    // divide whatever the previous run left behind.
    expect(listCourseSeeds).toHaveBeenCalledWith(COURSE_490);
    expect(exportApprovedCourseSeeds).toHaveBeenCalledWith(COURSE_490);
    expect(prepareTrainingSplit).toHaveBeenCalledWith(COURSE_490);

    expect(listCourseSeeds.mock.invocationCallOrder[0]).toBeLessThan(
      exportApprovedCourseSeeds.mock.invocationCallOrder[0],
    );
    expect(exportApprovedCourseSeeds.mock.invocationCallOrder[0]).toBeLessThan(
      prepareTrainingSplit.mock.invocationCallOrder[0],
    );
  });

  it('records the split counts and a relative dataset reference', async () => {
    const { preparation } = await prepareTrainingDataForRequest(COURSE_490);

    expect(preparation.trainExamples).toBe(38);
    expect(preparation.validationExamples).toBe(4);
    expect(preparation.sourceApprovedExampleCount).toBe(42);
    expect(preparation.splitSeed).toBe(360);
    expect(preparation.datasetRef).toBe(`exports/${COURSE_490}`);
    // An absolute path would embed a machine layout into a record the
    // professor UI also reads.
    expect(preparation.datasetRef.startsWith('/')).toBe(false);
  });

  it('moves the request to preparing, not training', async () => {
    await prepareTrainingDataForRequest(COURSE_490);

    const [courseId, patch] = updateCourseModelRequest.mock.calls[0];
    expect(courseId).toBe(COURSE_490);
    // Nothing is training: a dataset exists and no job has been submitted.
    expect(patch.status).toBe('preparing');
    expect(patch.preparation.trainExamples).toBe(38);
  });

  it('clears an error left by a previous attempt', async () => {
    await prepareTrainingDataForRequest(COURSE_490);
    expect(updateCourseModelRequest.mock.calls[0][1].preparationError).toBe('');
  });

  it('re-counts approved examples instead of trusting the request', async () => {
    // The professor asked when 42 were approved; 31 remain approved now.
    listCourseSeeds.mockResolvedValue({
      seeds: [...approvedSeeds(31), { id: 'r1', question: 'q', answer: 'a', reviewStatus: 'rejected' }],
    });

    const { preparation } = await prepareTrainingDataForRequest(COURSE_490);
    expect(preparation.sourceApprovedExampleCount).toBe(31);
  });
});

describe('insufficient approved examples', () => {
  it('refuses before exporting anything', async () => {
    listCourseSeeds.mockResolvedValue({ seeds: approvedSeeds(12) });

    await expect(prepareTrainingDataForRequest(COURSE_490)).rejects.toBeInstanceOf(
      InsufficientApprovedExamplesError,
    );

    expect(exportApprovedCourseSeeds).not.toHaveBeenCalled();
    expect(prepareTrainingSplit).not.toHaveBeenCalled();
  });

  it('leaves the request retryable with the reason recorded', async () => {
    listCourseSeeds.mockResolvedValue({ seeds: approvedSeeds(12) });

    await expect(prepareTrainingDataForRequest(COURSE_490)).rejects.toThrow();

    const [, patch] = updateCourseModelRequest.mock.calls[0];
    // `failed` is terminal and would unlock a fresh professor request over a
    // data-preparation problem.
    expect(patch.status).toBe('requested');
    expect(patch.preparationError).toMatch(/12 approved/);
  });

  it('honours an explicit minimum', async () => {
    listCourseSeeds.mockResolvedValue({ seeds: approvedSeeds(12) });

    await expect(
      prepareTrainingDataForRequest(COURSE_490, { minimumApproved: 10 }),
    ).resolves.toBeTruthy();
  });
});

describe('failure and retry', () => {
  it('returns the request to requested when the export fails', async () => {
    exportApprovedCourseSeeds.mockRejectedValue(new Error('export blew up'));

    await expect(prepareTrainingDataForRequest(COURSE_490)).rejects.toThrow(
      'export blew up',
    );

    const [, patch] = updateCourseModelRequest.mock.calls[0];
    expect(patch.status).toBe('requested');
    expect(patch.preparationError).toBe('export blew up');
    expect(patch.preparation).toBeUndefined();
  });

  it('returns the request to requested when the split fails', async () => {
    prepareTrainingSplit.mockRejectedValue(new Error('no approved export found'));

    await expect(prepareTrainingDataForRequest(COURSE_490)).rejects.toThrow();
    expect(updateCourseModelRequest.mock.calls[0][1].status).toBe('requested');
  });

  it('still reports the original failure when recording it also fails', async () => {
    exportApprovedCourseSeeds.mockRejectedValue(new Error('export blew up'));
    updateCourseModelRequest.mockRejectedValue(new Error('the database is down'));

    await expect(prepareTrainingDataForRequest(COURSE_490)).rejects.toThrow(
      'export blew up',
    );
  });

  it('succeeds on retry after a transient failure', async () => {
    exportApprovedCourseSeeds.mockRejectedValueOnce(new Error('transient'));

    await expect(prepareTrainingDataForRequest(COURSE_490)).rejects.toThrow();
    await expect(prepareTrainingDataForRequest(COURSE_490)).resolves.toMatchObject({
      preparation: { trainExamples: 38 },
    });

    expect(updateCourseModelRequest.mock.calls.at(-1)?.[1].status).toBe('preparing');
  });
});

describe('course isolation', () => {
  it('reads, exports, splits and records against one course only', async () => {
    await prepareTrainingDataForRequest(COURSE_490);

    for (const mock of [
      listCourseSeeds,
      exportApprovedCourseSeeds,
      prepareTrainingSplit,
      updateCourseModelRequest,
    ]) {
      for (const call of mock.mock.calls) {
        expect(call[0]).toBe(COURSE_490);
        expect(call[0]).not.toBe(COURSE_360);
      }
    }
  });

  it('scopes the dataset reference to the course being prepared', () => {
    expect(datasetRefForCourse(COURSE_490)).toBe(`exports/${COURSE_490}`);
    expect(datasetRefForCourse(COURSE_360)).toBe(`exports/${COURSE_360}`);
    expect(datasetRefForCourse(COURSE_490)).not.toBe(datasetRefForCourse(COURSE_360));
  });

  it('refuses an unsafe course id before touching anything', async () => {
    await expect(prepareTrainingDataForRequest('Bad_Id')).rejects.toThrow();

    expect(listCourseSeeds).not.toHaveBeenCalled();
    expect(exportApprovedCourseSeeds).not.toHaveBeenCalled();
    expect(updateCourseModelRequest).not.toHaveBeenCalled();
  });
});

describe('scope', () => {
  it('imports nothing that could train, promote, or register a model', async () => {
    const fs = await import('node:fs');
    const source = fs.readFileSync('src/lib/prepareTrainingData.ts', 'utf8');

    // Check the imports, not the prose — the docblock legitimately says what
    // this module does *not* do.
    const imports = source
      .split('\n')
      .filter((line) => /^\s*import\b/.test(line) || /^\s*}\s*from\s*'/.test(line))
      .join('\n');

    // The registry records artifacts that exist; preparation produces none.
    expect(imports).not.toMatch(/courseModelDb/);
    expect(imports).not.toMatch(/adminApi/);

    // Only the two existing endpoints do the export and split work.
    expect(source).toContain('exportApprovedCourseSeeds');
    expect(source).toContain('prepareTrainingSplit');
    // No second split implementation.
    expect(source).not.toMatch(/validationFraction|shuffle|Math\.random/);
  });
});
