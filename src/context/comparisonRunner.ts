import {
  generateBaseModel,
  generateFineTuned,
  generateFineTunedRag,
  generateRag,
} from '../lib/api';
import { toUserMessage } from '../lib/errorMessages';
import type { Role } from './role';
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
 *
 * Stored answers
 * --------------
 * When staff have prepared a question, an approach's answer may come back from
 * the classroom answer cache in a few milliseconds. Three of the four requests
 * leave together, so three stored answers would land in the same instant and
 * the cards would all fill at once. A stored answer is therefore shown no
 * earlier than its card's `CACHED_REVEAL_MS` after the question was asked,
 * which fills the cards in order. An answer generated for this request is
 * never held back: it is shown the moment it arrives. Nothing else changes,
 * no request is added, and nothing on the card says where the answer came from.
 */

export type ApproachState =
  | { status: 'idle' }
  | { status: 'loading' }
  | {
      status: 'success';
      text: string;
      sources: string[];
      /** Which model (and version) answered. Set only when the caller asked for it. */
      model?: string;
    }
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

/**
 * The earliest a stored answer is shown, in milliseconds after the question
 * was asked, per card and in card order. Short on purpose: a prepared
 * comparison is complete in under two seconds.
 */
export const CACHED_REVEAL_MS: Record<ModelKey, number> = {
  base: 400,
  rag: 800,
  fineTuned: 1200,
  fineTunedRag: 1600,
};

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

export interface ConditionReporter {
  /** One condition settled, successfully or not. Called once per condition. */
  onSettled: (key: ModelKey, state: ApproachState, response: ComparisonRunResponse) => void;
}

export interface RunOptions {
  /**
   * Whose wording a failed condition is reported in. The student pages (and
   * an administrator previewing them) use `student`; the staff question tab
   * uses the signed-in staff role.
   */
  audience?: Role;
  /** Report which model and version answered each condition (staff testing). */
  includeModel?: boolean;
}

function answeredBy(model: string, version?: string | null): string {
  return version ? `${model} · ${version}` : model;
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
  { audience = 'student', includeModel = false }: RunOptions = {},
): Promise<Record<ModelKey, ComparisonRunResponse>> {
  const responses: Record<ModelKey, ComparisonRunResponse> = {
    base: { text: '', error: null, sources: [] },
    rag: { text: '', error: null, sources: [] },
    fineTuned: { text: '', error: null, sources: [] },
    fineTunedRag: { text: '', error: null, sources: [] },
  };

  interface Settled {
    answer: string;
    sources?: string[];
    /** The model's name as the backend reported it. */
    model?: string | null;
    modelVersion?: string | null;
    engine?: string | null;
    /** Served from the classroom answer cache rather than generated now. */
    cached?: boolean;
    cacheEntryId?: string | null;
  }

  const askedAt = Date.now();

  const settle = async (key: ModelKey, request: () => Promise<Settled>) => {
    try {
      const { answer, sources = [], model, modelVersion, engine, cached, cacheEntryId } =
        await request();
      if (cached === true) {
        // A stored answer: keep the cards filling in order (see the header).
        const remaining = CACHED_REVEAL_MS[key] - (Date.now() - askedAt);
        if (remaining > 0) {
          await delay(remaining);
        }
      }
      // The provenance is always recorded on the response — it is what a
      // rating stores — and only shown in the card when the caller asked.
      responses[key] = {
        text: answer,
        error: null,
        sources,
        model: model ?? null,
        modelVersion: modelVersion ?? null,
        engine: engine ?? null,
        ...(cacheEntryId ? { cacheEntryId, cached: cached === true } : {}),
      };
      reporter.onSettled(
        key,
        includeModel && model
          ? { status: 'success', text: answer, sources, model: answeredBy(model, modelVersion) }
          : { status: 'success', text: answer, sources },
        responses[key],
      );
    } catch (error) {
      const message = toUserMessage(error, { audience, context: 'model-response' }).message;
      responses[key] = {
        text: '',
        error: message,
        sources: [],
        model: null,
        modelVersion: null,
        engine: null,
      };
      reporter.onSettled(key, { status: 'error', message }, responses[key]);
    }
  };

  const comparisonId = generateRunId();
  const base = () =>
    settle('base', async () => {
      const result = await generateBaseModel(courseId, question, comparisonId);
      return {
        answer: result.answer,
        model: result.model,
        cached: result.cached,
        cacheEntryId: result.cacheEntryId,
      };
    });
  const rag = () =>
    settle('rag', async () => {
      const result = await generateRag(courseId, question, undefined, comparisonId);
      return {
        answer: result.answer,
        sources: formatRagSourceLabels(result.sources),
        model: result.model,
        cached: result.cached,
        cacheEntryId: result.cacheEntryId,
      };
    });
  const fineTuned = () =>
    settle('fineTuned', async () => {
      const result = await generateFineTuned(courseId, question, comparisonId);
      return {
        answer: result.answer,
        model: result.model,
        modelVersion: result.modelVersion,
        engine: result.engine,
        cached: result.cached,
        cacheEntryId: result.cacheEntryId,
      };
    });
  const fineTunedRag = () =>
    settle('fineTunedRag', async () => {
      const result = await generateFineTunedRag(courseId, question, undefined, comparisonId);
      return {
        answer: result.answer,
        sources: formatRagSourceLabels(result.sources),
        model: result.model,
        modelVersion: result.modelVersion,
        engine: result.engine,
        cached: result.cached,
        cacheEntryId: result.cacheEntryId,
      };
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
