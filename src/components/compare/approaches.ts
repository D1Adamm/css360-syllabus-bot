import type { ModelKey } from '../../types';

/**
 * The four comparison approaches, named for the people using them.
 *
 * The internal keys stay exactly as they were — `base`, `rag`, `fineTuned`,
 * `fineTunedRag` — so stored evaluations, backend routes and aggregation are
 * untouched. Only what a person reads changes.
 *
 * "RAG" is jargon that describes an implementation, not an idea a student needs.
 * "Syllabus-Aware" describes what actually differs between these answers.
 */

export interface ApproachCopy {
  key: ModelKey;
  label: string;
  description: string;
}

export const APPROACHES: ApproachCopy[] = [
  {
    key: 'base',
    label: 'Base Model',
    description: 'General model, no course context',
  },
  {
    key: 'rag',
    label: 'Syllabus-Aware',
    description: 'Uses information from your syllabus',
  },
  {
    key: 'fineTuned',
    label: 'Course-Trained',
    description: 'Learned from approved course examples',
  },
  {
    key: 'fineTunedRag',
    label: 'Course-Trained + Syllabus',
    description: 'Combines course examples with syllabus context',
  },
];

const BY_KEY = new Map(APPROACHES.map((approach) => [approach.key, approach]));

export function approachLabel(key: ModelKey): string {
  return BY_KEY.get(key)?.label ?? key;
}

export function approachDescription(key: ModelKey): string {
  return BY_KEY.get(key)?.description ?? '';
}
