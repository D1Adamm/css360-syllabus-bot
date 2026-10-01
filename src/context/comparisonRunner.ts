import {
  generateBaseModel,
  generateFineTuned,
  generateFineTunedRag,
  generateRag,
} from '../lib/api';
import { toUserMessage } from '../lib/errorMessages';
import type { ModelKey } from '../types';
import { formatRagSourceLabels } from '../utils/ragSourceLabels';
import type { ComparisonRunResponse } from './comparisonRun';

/**
 * The four requests one comparison makes, independent of any page.
 *
 * Lives outside React so a comparison belongs to the application, not to the
 * Compare page: the page can unmount while a student goes to Contribute and
 * the requests carry on, reporting each answer as it arrives.
 *
 * The control flow is the original compare page's, unchanged:
 *
 *   - Base and RAG run strictly in sequence — RAG only starts once Base has
 *     settled — because the backend serialises them on one CPU-bound model.
 *   - The two fine-tuned requests overlap with that chain.
 *   - A failure in one request never cancels another; RAG still runs when
 *     Base fails, and every answer is reported the moment it arrives.
 *   - All four carry one comparison id, so the backend's generation queue
 *     finishes this comparison before ones that began after it.
 */

export type ApproachState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; text: string; sources: string[] }
  | { status: 'error'; message: string };

export type ApproachStates = Record<ModelKey, ApproachState>;

export const IDLE_STATES: ApproachStates = {
  base: { status: 'idle' },
  rag: { status: 'idle' },
  fineTuned: { status: 'idle' },
  fineTunedRag: { status: 'idle' },
};

export const LOADING_STATES: ApproachStates = {
  base: { status: 'loading' },
  rag: { status: 'loading' },
  fineTuned: { status: 'loading' },
  fineTunedRag: { status: 'loading' },
};

export interface ConditionReporter {
  /** One condition settled, successfully or not. Called once per condition. */
  onSettled: (key: ModelKey, state: ApproachState, response: ComparisonRunResponse) => void;
}

function studentMessage(error: unknown): string {
  return toUserMessage(error, { audience: 'student', context: 'model-response' }).message;
}

export function generateRunId(): string {
  return `run-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * Ask all four conditions and resolve with every response once all have
 * settled. Never rejects: each condition's failure is reported as its own
 * state and carried in its response.
 */
export async function runFourConditions(
  courseId: string,
  question: string,
  reporter: ConditionReporter,
): Promise<Record<ModelKey, ComparisonRunResponse>> {
  const responses: Record<ModelKey, ComparisonRunResponse> = {
    base: { text: '', error: null, sources: [] },
    rag: { text: '', error: null, sources: [] },
    fineTuned: { text: '', error: null, sources: [] },
    fineTunedRag: { text: '', error: null, sources: [] },
  };

  const settle = async (
    key: ModelKey,
    request: () => Promise<{ answer: string; sources?: string[] }>,
  ) => {
    try {
      const { answer, sources = [] } = await request();
      responses[key] = { text: answer, error: null, sources };
      reporter.onSettled(key, { status: 'success', text: answer, sources }, responses[key]);
    } catch (error) {
      const message = studentMessage(error);
      responses[key] = { text: '', error: message, sources: [] };
      reporter.onSettled(key, { status: 'error', message }, responses[key]);
    }
  };

  const comparisonId = generateRunId();
  const base = () =>
    settle('base', async () => ({
      answer: (await generateBaseModel(courseId, question, comparisonId)).answer,
    }));
  const rag = () =>
    settle('rag', async () => {
      const result = await generateRag(courseId, question, undefined, comparisonId);
      return { answer: result.answer, sources: formatRagSourceLabels(result.sources) };
    });
  const fineTuned = () =>
    settle('fineTuned', async () => ({
      answer: (await generateFineTuned(courseId, question, comparisonId)).answer,
    }));
  const fineTunedRag = () =>
    settle('fineTunedRag', async () => {
      const result = await generateFineTunedRag(courseId, question, undefined, comparisonId);
      return { answer: result.answer, sources: formatRagSourceLabels(result.sources) };
    });

  await Promise.all([
    (async () => {
      await base();
      await rag();
    })(),
    fineTuned(),
    fineTunedRag(),
  ]);
  return responses;
}
