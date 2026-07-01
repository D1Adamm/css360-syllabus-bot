import { describe, expect, it } from 'vitest';
import comparisonData from '../data/comparisonData.json';
import type { ComparisonRecord } from '../types';
import {
  SIMULATED_UNAVAILABLE_MESSAGE,
  allCardsReferToSameActiveQuestion,
  getComparisonModeFromCustomSubmit,
  resolveSimulatedRecord,
  shouldShowSimulatedFineTunedResponses,
} from './comparisonPageState';
import { findBestComparisonMatch } from './comparisonUtils';

const records = comparisonData as ComparisonRecord[];

const DISCORD_GRADE_QUESTION = 'Can grade questions be discussed in Discord?';

describe('findBestComparisonMatch', () => {
  it('does not match passing grade questions to the Discord grade question', () => {
    const match = findBestComparisonMatch('What is the passing grade?', records);

    expect(match).toBeNull();
  });

  it('does not match passing grade questions without a question mark', () => {
    const match = findBestComparisonMatch('What is the passing grade', records);

    expect(match).toBeNull();
  });

  it('matches genuinely equivalent predefined questions', () => {
    const match = findBestComparisonMatch(DISCORD_GRADE_QUESTION, records);

    expect(match?.matchedQuestion).toBe(DISCORD_GRADE_QUESTION);
    expect(match?.recordId).toBe('comparison-013');
  });
});

describe('comparison page state', () => {
  const selectedRecord = records[0];
  const discordRecord = records.find((record) => record.id === 'comparison-013');

  it('enters custom-unmatched mode when no predefined question matches', () => {
    expect(getComparisonModeFromCustomSubmit(null)).toBe('custom-unmatched');
    expect(shouldShowSimulatedFineTunedResponses('custom-unmatched')).toBe(false);
  });

  it('clears simulated responses when returning to predefined mode', () => {
    const simulatedRecord = resolveSimulatedRecord(
      'predefined',
      selectedRecord,
      discordRecord ?? null,
    );

    expect(simulatedRecord?.id).toBe(selectedRecord.id);
    expect(
      allCardsReferToSameActiveQuestion(
        'predefined',
        selectedRecord.question,
        simulatedRecord,
      ),
    ).toBe(true);
  });

  it('never silently shows unrelated simulated responses for custom questions', () => {
    const customQuestion = 'What is the passing grade?';
    const simulatedRecord = resolveSimulatedRecord(
      'custom-unmatched',
      selectedRecord,
      discordRecord ?? null,
    );

    expect(simulatedRecord).toBeNull();
    expect(
      allCardsReferToSameActiveQuestion('custom-unmatched', customQuestion, simulatedRecord),
    ).toBe(true);
    expect(SIMULATED_UNAVAILABLE_MESSAGE).toContain('No simulated fine-tuned response');
  });
});
