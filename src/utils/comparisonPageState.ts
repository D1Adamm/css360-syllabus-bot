import type { ComparisonRecord } from '../types';
import type { MatchResult } from './comparisonUtils';

export type ComparisonMode = 'predefined' | 'custom-matched' | 'custom-unmatched';

export const SIMULATED_UNAVAILABLE_MESSAGE =
  'No simulated fine-tuned response is available for this custom question.';

export function getComparisonModeFromCustomSubmit(
  match: MatchResult | null,
): ComparisonMode {
  return match ? 'custom-matched' : 'custom-unmatched';
}

export function shouldShowSimulatedFineTunedResponses(mode: ComparisonMode): boolean {
  return mode === 'predefined' || mode === 'custom-matched';
}

export function resolveSimulatedRecord(
  mode: ComparisonMode,
  selectedRecord: ComparisonRecord,
  matchedRecord: ComparisonRecord | null,
): ComparisonRecord | null {
  if (!shouldShowSimulatedFineTunedResponses(mode)) {
    return null;
  }

  if (mode === 'custom-matched' && matchedRecord) {
    return matchedRecord;
  }

  return selectedRecord;
}

export function allCardsReferToSameActiveQuestion(
  mode: ComparisonMode,
  activeQuestion: string,
  simulatedRecord: ComparisonRecord | null,
): boolean {
  if (mode === 'custom-unmatched') {
    return simulatedRecord === null;
  }

  if (mode === 'predefined') {
    return simulatedRecord?.question === activeQuestion;
  }

  return simulatedRecord !== null;
}
