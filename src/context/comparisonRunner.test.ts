import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

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

import { snapshotOf } from './comparisonRun';
import { CACHED_REVEAL_MS, runFourConditions } from './comparisonRunner';

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

describe('runFourConditions stored answers', () => {
  /*
   * A prepared question's answers come back from the classroom answer cache in
   * milliseconds, and three of the four requests leave together. Without
   * pacing, three cards would fill in the same instant. A stored answer is
   * shown no earlier than its card's reveal time; a generated one is never
   * held back.
   */
  const stored = (answer: string, extra: Record<string, unknown> = {}) => ({
    answer,
    model: 'm',
    sources: [],
    cached: true,
    cacheEntryId: `entry-${answer}`,
    ...extra,
  });

  function track() {
    const shown: Array<{ key: string; at: number }> = [];
    const started = Date.now();
    const reporter = {
      onSettled: vi.fn((key: string, _state: { status: string; text?: string }) => {
        shown.push({ key, at: Date.now() - started });
      }),
    };
    return { shown, reporter };
  }

  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('fills the four cards in order, a short step apart, when every answer is stored', async () => {
    api.generateBaseModel.mockResolvedValue(stored('base'));
    api.generateRag.mockResolvedValue(stored('rag'));
    api.generateFineTuned.mockResolvedValue(stored('ft'));
    api.generateFineTunedRag.mockResolvedValue(stored('ftrag'));
    const { shown, reporter } = track();

    const run = runFourConditions('css-360', 'When is the exam?', reporter);
    await vi.advanceTimersByTimeAsync(0);
    // All four arrived at once; none is shown yet.
    expect(shown).toEqual([]);

    await vi.advanceTimersByTimeAsync(CACHED_REVEAL_MS.fineTunedRag);
    await run;

    expect(shown).toEqual([
      { key: 'base', at: CACHED_REVEAL_MS.base },
      { key: 'rag', at: CACHED_REVEAL_MS.rag },
      { key: 'fineTuned', at: CACHED_REVEAL_MS.fineTuned },
      { key: 'fineTunedRag', at: CACHED_REVEAL_MS.fineTunedRag },
    ]);
    // Each answer stays with its own approach.
    expect(reporter.onSettled.mock.calls.map(([key, state]) => [key, state.text])).toEqual([
      ['base', 'base'],
      ['rag', 'rag'],
      ['fineTuned', 'ft'],
      ['fineTunedRag', 'ftrag'],
    ]);
    // A prepared comparison is complete in well under two seconds, with no extra request.
    expect(CACHED_REVEAL_MS.fineTunedRag).toBeLessThanOrEqual(2000);
    expect(api.generateBaseModel).toHaveBeenCalledTimes(1);
    expect(api.generateRag).toHaveBeenCalledTimes(1);
    expect(api.generateFineTuned).toHaveBeenCalledTimes(1);
    expect(api.generateFineTunedRag).toHaveBeenCalledTimes(1);
  });

  it('never holds back an answer that was generated for this request', async () => {
    api.generateBaseModel.mockResolvedValue({ answer: 'base', model: 'm' });
    api.generateRag.mockResolvedValue({ answer: 'rag', model: 'm', sources: [] });
    api.generateFineTuned.mockResolvedValue({ answer: 'ft', model: 'm', cached: false });
    api.generateFineTunedRag.mockResolvedValue({ answer: 'ftrag', model: 'm', sources: [] });
    const { shown, reporter } = track();

    const run = runFourConditions('css-360', 'When is the exam?', reporter);
    await vi.advanceTimersByTimeAsync(0);
    await run;

    expect(shown.map((entry) => entry.at)).toEqual([0, 0, 0, 0]);
  });

  it('shows stored answers on schedule and generated ones whenever they arrive', async () => {
    // Base and Fine-Tuned + RAG are prepared; RAG and Fine-Tuned generate.
    const later = <T,>(value: T, ms: number) =>
      new Promise<T>((resolve) => {
        setTimeout(() => resolve(value), ms);
      });
    api.generateBaseModel.mockResolvedValue(stored('base'));
    api.generateRag.mockImplementation(() => later({ answer: 'rag', model: 'm', sources: [] }, 9000));
    api.generateFineTuned.mockImplementation(() => later({ answer: 'ft', model: 'm' }, 300));
    api.generateFineTunedRag.mockResolvedValue(stored('ftrag'));
    const { shown, reporter } = track();

    const run = runFourConditions('css-360', 'When is the exam?', reporter);
    await vi.advanceTimersByTimeAsync(10000);
    await run;

    expect(shown).toEqual([
      // Generated in 300 ms: shown at once, ahead of the stored Base card.
      { key: 'fineTuned', at: 300 },
      { key: 'base', at: CACHED_REVEAL_MS.base },
      { key: 'fineTunedRag', at: CACHED_REVEAL_MS.fineTunedRag },
      // RAG starts after Base is shown, as it always has, and takes 9 s.
      { key: 'rag', at: CACHED_REVEAL_MS.base + 9000 },
    ]);
  });

  it('does not delay a stored answer that arrives after its reveal time', async () => {
    const later = <T,>(value: T, ms: number) =>
      new Promise<T>((resolve) => {
        setTimeout(() => resolve(value), ms);
      });
    api.generateBaseModel.mockImplementation(() => later({ answer: 'base', model: 'm' }, 5000));
    api.generateRag.mockResolvedValue(stored('rag'));
    api.generateFineTuned.mockResolvedValue({ answer: 'ft', model: 'm' });
    api.generateFineTunedRag.mockResolvedValue({ answer: 'ftrag', model: 'm', sources: [] });
    const { shown, reporter } = track();

    const run = runFourConditions('css-360', 'When is the exam?', reporter);
    await vi.advanceTimersByTimeAsync(6000);
    await run;

    // RAG's stored answer came back at 5 s, long past its 0.8 s mark.
    expect(shown.find((entry) => entry.key === 'rag')?.at).toBe(5000);
  });

  it('records the stored answer on the run for the rating, and nothing on an ordinary one', async () => {
    api.generateBaseModel.mockResolvedValue(stored('base'));
    api.generateRag.mockResolvedValue({ answer: 'rag', model: 'm', sources: [], cached: false, cacheEntryId: 'new-row' });
    api.generateFineTuned.mockResolvedValue({ answer: 'ft', model: 'm', cached: false, cacheEntryId: null });
    api.generateFineTunedRag.mockRejectedValue(new Error('down'));

    const run = runFourConditions('css-360', 'When is the exam?', { onSettled: vi.fn() });
    await vi.advanceTimersByTimeAsync(2000);
    const responses = await run;

    expect(responses.base).toMatchObject({ cached: true, cacheEntryId: 'entry-base' });
    // Generated now and stored: the row's id, and not served from the store.
    expect(responses.rag).toMatchObject({ cached: false, cacheEntryId: 'new-row' });
    expect(responses.fineTuned).not.toHaveProperty('cacheEntryId');
    expect(responses.fineTuned).not.toHaveProperty('cached');
    expect(responses.fineTunedRag).not.toHaveProperty('cacheEntryId');

    expect(snapshotOf(responses.base)).toMatchObject({ cached: true, cacheEntryId: 'entry-base' });
    expect(Object.keys(snapshotOf(responses.fineTuned)).sort()).toEqual(
      ['engine', 'error', 'model', 'modelVersion', 'sources', 'text'],
    );
  });

  it('never tells the card where the answer came from', async () => {
    api.generateBaseModel.mockResolvedValue(stored('base'));
    api.generateRag.mockResolvedValue(stored('rag'));
    api.generateFineTuned.mockResolvedValue(stored('ft'));
    api.generateFineTunedRag.mockResolvedValue(stored('ftrag'));
    const reporter = { onSettled: vi.fn() };

    const run = runFourConditions('css-360', 'When is the exam?', reporter, { includeModel: true });
    await vi.advanceTimersByTimeAsync(2000);
    await run;

    for (const [, state] of reporter.onSettled.mock.calls) {
      expect(Object.keys(state).sort()).toEqual(['model', 'sources', 'status', 'text']);
      expect(JSON.stringify(state)).not.toMatch(/cach/i);
    }
  });
});
