import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({
  generateBaseModel: vi.fn(),
  generateRag: vi.fn(),
  generateFineTuned: vi.fn(),
  generateFineTunedRag: vi.fn(),
}));

vi.mock('../lib/api', () => api);

import { runFourConditions } from './comparisonRunner';

describe('runFourConditions comparison id', () => {
  beforeEach(() => {
    api.generateBaseModel.mockResolvedValue({ answer: 'base' });
    api.generateRag.mockResolvedValue({ answer: 'rag', sources: [] });
    api.generateFineTuned.mockResolvedValue({ answer: 'ft' });
    api.generateFineTunedRag.mockResolvedValue({ answer: 'ftrag', sources: [] });
  });

  it('sends one id with all four requests of a comparison, and a new id for the next', async () => {
    const reporter = { onSettled: vi.fn() };
    await runFourConditions('css-360', 'When is the exam?', reporter);
    await runFourConditions('css-360', 'When is the exam?', reporter);

    const first = [
      api.generateBaseModel.mock.calls[0][2],
      api.generateRag.mock.calls[0][3],
      api.generateFineTuned.mock.calls[0][2],
      api.generateFineTunedRag.mock.calls[0][3],
    ];
    expect(new Set(first).size).toBe(1);
    expect(first[0]).toMatch(/^run-/);
    expect(api.generateBaseModel.mock.calls[1][2]).not.toBe(first[0]);
    expect(reporter.onSettled).toHaveBeenCalledTimes(8);
  });
});
