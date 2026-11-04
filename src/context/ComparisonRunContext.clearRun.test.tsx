/** @vitest-environment jsdom */
import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

const api = vi.hoisted(() => ({
  generateBaseModel: vi.fn(),
  generateRag: vi.fn(),
  generateFineTuned: vi.fn(),
  generateFineTunedRag: vi.fn(),
}));

vi.mock('../lib/api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../lib/api')>()),
  ...api,
}));

import { ComparisonRunProvider, useComparisonRunStore } from './ComparisonRunContext';

const COURSE = 'css-360-winter-2026-a7rp';

function Probe() {
  const store = useComparisonRunStore();
  const active = store.getActive(COURSE);
  const run = store.getRun(COURSE);
  return (
    <div>
      <button type="button" onClick={() => store.startComparison(COURSE, 'When?', null)}>
        ask
      </button>
      <button type="button" onClick={() => store.clearRun(COURSE)}>
        clear
      </button>
      <button type="button" onClick={() => run && store.markRated(COURSE, run.runId, 'e1')}>
        rate
      </button>
      <button type="button" onClick={() => store.markRated(COURSE, 'run-elsewhere', 'e2')}>
        rate other
      </button>
      <output data-testid="active">{active ? (active.isRunning ? 'running' : 'settled') : 'none'}</output>
      <output data-testid="run">{run ? `${run.runId}:${run.ratedAt ? 'rated' : 'unrated'}` : 'none'}</output>
    </div>
  );
}

beforeEach(() => {
  window.sessionStorage.clear();
  api.generateBaseModel.mockResolvedValue({ answer: 'b', model: 'm' });
  api.generateRag.mockResolvedValue({ answer: 'r', model: 'm', sources: [] });
  api.generateFineTuned.mockResolvedValue({ answer: 'f', model: 'm' });
  api.generateFineTunedRag.mockResolvedValue({ answer: 'fr', model: 'm', sources: [] });
});

afterEach(() => {
  cleanup();
});

async function askAndSettle() {
  render(
    <ComparisonRunProvider>
      <Probe />
    </ComparisonRunProvider>,
  );
  await act(async () => {
    screen.getByRole('button', { name: 'ask' }).click();
  });
  await waitFor(() => {
    expect(screen.getByTestId('active')).toHaveTextContent('settled');
    expect(screen.getByTestId('run')).toHaveTextContent(':unrated');
  });
}

describe('ComparisonRunProvider', () => {
  it('clearRun forgets the stored run and the settled comparison together', async () => {
    await askAndSettle();

    await act(async () => {
      screen.getByRole('button', { name: 'clear' }).click();
    });

    // Compare would otherwise keep showing the settled answers with an
    // "Evaluate these responses" link that led to "Nothing to evaluate yet".
    expect(screen.getByTestId('active')).toHaveTextContent('none');
    expect(screen.getByTestId('run')).toHaveTextContent('none');
    expect(window.sessionStorage.getItem(`sml.run.${COURSE}`)).toBeNull();
  });

  it('markRated marks only the run it was asked about', async () => {
    await askAndSettle();

    await act(async () => {
      screen.getByRole('button', { name: 'rate other' }).click();
    });
    expect(screen.getByTestId('run')).toHaveTextContent(':unrated');

    await act(async () => {
      screen.getByRole('button', { name: 'rate' }).click();
    });
    expect(screen.getByTestId('run')).toHaveTextContent(':rated');
    const stored = JSON.parse(window.sessionStorage.getItem(`sml.run.${COURSE}`) ?? 'null');
    expect(stored.evaluationId).toBe('e1');
  });
});
