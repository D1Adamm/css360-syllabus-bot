import { beforeEach, describe, expect, it, vi } from 'vitest';

const http = vi.hoisted(() => ({
  getJson: vi.fn(),
  sendJson: vi.fn(),
}));

vi.mock('./httpClient', () => http);

import {
  fetchPreparationStatus,
  filterQuestions,
  preparationPresentation,
  preparationSummary,
  primaryAction,
  savePreparationSelection,
  startPreparation,
  stopPreparation,
} from './classroomPreparation';
import { eightyOfOneTwenty, preparation, question } from '../test/preparationFixtures';

const COURSE = 'css-360-winter-2026-a7rp';

describe('classroom preparation requests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    http.getJson.mockResolvedValue(preparation());
    http.sendJson.mockResolvedValue(preparation());
  });

  it('uses the course-scoped paths, relative to the API base', async () => {
    await fetchPreparationStatus(COURSE);
    await savePreparationSelection(COURSE, ['s1', 's2']);
    await startPreparation(COURSE);
    await stopPreparation(COURSE);

    expect(http.getJson.mock.calls[0][0]).toBe(`/courses/${COURSE}/classroom-preparation`);
    expect(http.sendJson.mock.calls.map((call) => [call[0], call[1], call[2]])).toEqual([
      ['PUT', `/courses/${COURSE}/classroom-preparation/questions`, { seedIds: ['s1', 's2'] }],
      ['POST', `/courses/${COURSE}/classroom-preparation/start`, undefined],
      ['POST', `/courses/${COURSE}/classroom-preparation/stop`, undefined],
    ]);
  });
});

describe('what a status is called', () => {
  it('names every state the backend can report', () => {
    expect(
      (
        ['not_started', 'preparing', 'stopping', 'ready', 'partially_ready', 'stopped', 'failed'] as const
      ).map((state) => preparationPresentation(state).label),
    ).toEqual([
      'Not started',
      'Preparing',
      'Stopping',
      'Ready',
      'Partially ready',
      'Stopped',
      'Needs attention',
    ]);
    expect(preparationPresentation('ready').tone).toBe('success');
    expect(preparationPresentation('failed').tone).toBe('danger');
  });

  it('summarises with the real counts', () => {
    expect(preparationSummary(eightyOfOneTwenty())).toBe('20 of 30 questions ready');
    // While a run is going, the finer-grained number is the one that moves.
    expect(preparationSummary(eightyOfOneTwenty({ state: 'preparing', running: true }))).toBe(
      '80 of 120 answers ready',
    );
    expect(
      preparationSummary(
        preparation({ counts: { ...preparation().counts, selectedQuestions: 0, expectedAnswers: 0 } }),
      ),
    ).toBe('No questions selected yet');
    expect(
      preparationSummary(
        preparation({
          counts: { ...preparation().counts, selectedQuestions: 1, fullyPreparedQuestions: 1 },
        }),
      ),
    ).toBe('1 of 1 question ready');
  });
});

describe('the one action', () => {
  it('offers Prepare answers when nothing is prepared', () => {
    expect(primaryAction(preparation())).toEqual({ kind: 'start', label: 'Prepare answers' });
  });

  it('offers Prepare remaining answers once some exist: resume and retry are the same request', () => {
    for (const state of ['partially_ready', 'stopped', 'failed'] as const) {
      expect(primaryAction(eightyOfOneTwenty({ state }))).toEqual({
        kind: 'start',
        label: 'Prepare remaining answers',
      });
    }
  });

  it('offers only Stop while running, and only a disabled Stopping while it stops', () => {
    expect(primaryAction(eightyOfOneTwenty({ state: 'preparing', running: true, canStart: false }))).toEqual({
      kind: 'stop',
      label: 'Stop preparation',
    });
    expect(
      primaryAction(
        eightyOfOneTwenty({ state: 'stopping', running: true, stopping: true, canStart: false }),
      ),
    ).toEqual({ kind: 'stopping', label: 'Stopping…' });
  });

  it('offers nothing when nothing can be prepared', () => {
    for (const reason of ['nothing_to_prepare', 'nothing_selected', 'disabled', 'gpu_mode', 'another_course']) {
      expect(
        primaryAction(preparation({ state: 'ready', canStart: false, startBlockedReason: reason })),
      ).toBeNull();
    }
  });
});

describe('filtering the question list', () => {
  const questions = preparation().questions;

  it('hides unselected behaviour examples until they are asked for', () => {
    const shown = filterQuestions(questions, { search: '', source: 'all', selectedOnly: false });
    expect(shown.map((q) => q.seedId)).toEqual(['s1', 's2', 's3']);
    expect(
      filterQuestions(questions, { search: '', source: 'authored', selectedOnly: false }).map((q) => q.seedId),
    ).toEqual(['s4']);
  });

  it('keeps a selected behaviour example visible so it can be deselected', () => {
    const withSelected = [...questions.slice(0, 3), question('s4', 'What is the weather today?', {
      source: 'authored',
      selected: true,
    })];
    expect(
      filterQuestions(withSelected, { search: '', source: 'all', selectedOnly: false }).map((q) => q.seedId),
    ).toEqual(['s1', 's2', 's3', 's4']);
  });

  it('filters by source, search text and selection together', () => {
    expect(
      filterQuestions(questions, { search: '', source: 'ai_generated', selectedOnly: false }).map((q) => q.seedId),
    ).toEqual(['s3']);
    expect(
      filterQuestions(questions, { search: '  LATE work ', source: 'all', selectedOnly: false }).map((q) => q.seedId),
    ).toEqual(['s2']);
    expect(
      filterQuestions(questions, { search: '', source: 'all', selectedOnly: true }).map((q) => q.seedId),
    ).toEqual(['s1', 's2']);
    expect(filterQuestions(questions, { search: 'exam', source: 'ai_generated', selectedOnly: false })).toEqual([]);
  });
});
