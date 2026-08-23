import { beforeEach, describe, expect, it, vi } from 'vitest';

const fetchMock = vi.fn();

vi.stubGlobal('fetch', fetchMock);

vi.stubEnv('VITE_API_BASE_URL', 'http://localhost:8000');

describe('seed review API helpers', () => {
  beforeEach(() => {
    fetchMock.mockReset();
  });

  it('lists course seeds from the course-scoped endpoint', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        courseId: 'css-360-winter-2026-a7rp',
        count: 1,
        seeds: [{ id: 's1', question: 'Q?', answer: 'A', reviewStatus: 'generated' }],
      }),
    });

    const { listCourseSeeds } = await import('./api');
    const result = await listCourseSeeds('css-360-winter-2026-a7rp');
    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8000/courses/css-360-winter-2026-a7rp/seeds',
      expect.objectContaining({ method: 'GET' }),
    );
    expect(result.count).toBe(1);
    expect(result.seeds[0].id).toBe('s1');
  });

  it('posts review updates for a seed', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        courseId: 'css-360-winter-2026-a7rp',
        seedId: 'seed-1',
        seed: { id: 'seed-1', reviewStatus: 'approved' },
      }),
    });

    const { reviewCourseSeed } = await import('./api');
    await reviewCourseSeed('css-360-winter-2026-a7rp', 'seed-1', {
      reviewStatus: 'approved',
    });
    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8000/courses/css-360-winter-2026-a7rp/seeds/seed-1/review',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ reviewStatus: 'approved' }),
      }),
    );
  });

  it('posts approved export requests', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        courseId: 'css-360-winter-2026-a7rp',
        summary: { approvedCount: 2 },
      }),
    });

    const { exportApprovedCourseSeeds } = await import('./api');
    const result = await exportApprovedCourseSeeds('css-360-winter-2026-a7rp');
    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8000/courses/css-360-winter-2026-a7rp/seeds/export-approved',
      expect.objectContaining({ method: 'POST' }),
    );
    expect(result.summary.approvedCount).toBe(2);
  });

  it('checks approved export status and prepares training splits', async () => {
    fetchMock
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          courseId: 'css-360-winter-2026-a7rp',
          exists: true,
          exportPath: 'data/exports/css-360-winter-2026-a7rp/approved-finetune.jsonl',
          exampleCount: 54,
          sourceFile: 'approved-finetune.jsonl',
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          courseId: 'css-360-winter-2026-a7rp',
          summary: { trainExamples: 48, validationExamples: 6 },
        }),
      });

    const { getApprovedExportStatus, prepareTrainingSplit } = await import('./api');
    const status = await getApprovedExportStatus('css-360-winter-2026-a7rp');
    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8000/courses/css-360-winter-2026-a7rp/seeds/approved-export-status',
      expect.objectContaining({ method: 'GET' }),
    );
    expect(status.exists).toBe(true);

    const split = await prepareTrainingSplit('css-360-winter-2026-a7rp');
    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8000/courses/css-360-winter-2026-a7rp/seeds/prepare-training-split',
      expect.objectContaining({ method: 'POST' }),
    );
    expect(split.summary.trainExamples).toBe(48);
    expect(split.summary.validationExamples).toBe(6);
  });
});
