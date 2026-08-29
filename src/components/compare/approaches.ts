import type { ModelKey } from '../../types';

/**
 * The four comparison approaches, named for the technique each one uses.
 *
 * The internal keys stay exactly as they were — `base`, `rag`, `fineTuned`,
 * `fineTunedRag` — so stored evaluations, backend routes and aggregation are
 * untouched. Only what a person reads changes.
 *
 * This is a research and teaching tool, so the names say which technical
 * approach produced an answer rather than describing it loosely. Everything
 * user-facing — compare cards, rating controls, charts, results tables —
 * reads its label from here, so the terminology cannot drift between the
 * student, professor and admin views.
 */

export interface ApproachCopy {
  key: ModelKey;
  label: string;
  description: string;
}

export const APPROACHES: ApproachCopy[] = [
  {
    key: 'base',
    label: 'Base',
    description: 'Base model, no course context',
  },
  {
    key: 'rag',
    label: 'RAG',
    description: 'Base model with retrieved syllabus context',
  },
  {
    key: 'fineTuned',
    label: 'Fine-Tuned',
    description: 'Course-specific fine-tuned model',
  },
  {
    key: 'fineTunedRag',
    label: 'Fine-Tuned + RAG',
    description: 'Fine-tuned model with retrieved syllabus context',
  },
];

const BY_KEY = new Map(APPROACHES.map((approach) => [approach.key, approach]));

export function approachLabel(key: ModelKey): string {
  return BY_KEY.get(key)?.label ?? key;
}

export function approachDescription(key: ModelKey): string {
  return BY_KEY.get(key)?.description ?? '';
}
