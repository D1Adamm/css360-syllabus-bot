import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({
  generateBaseModel: vi.fn(),
  generateRag: vi.fn(),
  generateFineTuned: vi.fn(),
  generateFineTunedRag: vi.fn(),
}));

// The runner's failure path renders a user-facing message, which needs the
// real `ApiError`; only the four generate calls are faked.
vi.mock('../lib/api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../lib/api')>()),
  ...api,
}));

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

describe('runFourConditions provenance', () => {
  /*
   * A rating stores what it rated. The run therefore carries, for every
   * condition, the model the backend said answered — and for the fine-tuned
   * conditions the course version and the engine (VM or classroom GPU). It did
   * not before 2026-11-03 (audit B8): the evaluations table named a winner
   * with no record of which version had won.
   */
  beforeEach(() => {
    api.generateBaseModel.mockResolvedValue({ answer: 'base', model: 'llama3.2:3b' });
    api.generateRag.mockResolvedValue({ answer: 'rag', model: 'llama3.2:3b', sources: [] });
    api.generateFineTuned.mockResolvedValue({
      answer: 'ft',
      model: 'css-360-ft',
      modelVersion: 'v2',
      engine: 'vm',
    });
    api.generateFineTunedRag.mockRejectedValue(new Error('down'));
  });

  it('records the model, version and engine on every response, and nulls on a failure', async () => {
    const responses = await runFourConditions('css-360', 'When is the exam?', {
      onSettled: vi.fn(),
    });

    expect(responses.base).toEqual({
      text: 'base',
      error: null,
      sources: [],
      model: 'llama3.2:3b',
      modelVersion: null,
      engine: null,
    });
    expect(responses.fineTuned).toEqual({
      text: 'ft',
      error: null,
      sources: [],
      model: 'css-360-ft',
      modelVersion: 'v2',
      engine: 'vm',
    });
    expect(responses.fineTunedRag.text).toBe('');
    expect(responses.fineTunedRag.error).toEqual(expect.any(String));
    expect(responses.fineTunedRag.model).toBeNull();
    expect(responses.fineTunedRag.modelVersion).toBeNull();
    expect(responses.fineTunedRag.engine).toBeNull();
  });

  it('still shows the model in the card only when asked (staff testing)', async () => {
    const quiet = { onSettled: vi.fn() };
    await runFourConditions('css-360', 'When is the exam?', quiet);
    const quietStates = quiet.onSettled.mock.calls.map(([, state]) => state);
    expect(quietStates.some((state) => 'model' in state)).toBe(false);

    const verbose = { onSettled: vi.fn() };
    await runFourConditions('css-360', 'When is the exam?', verbose, { includeModel: true });
    const fineTunedState = verbose.onSettled.mock.calls.find(([key]) => key === 'fineTuned')?.[1];
    expect(fineTunedState).toMatchObject({ status: 'success', model: 'css-360-ft · v2' });
  });
});
