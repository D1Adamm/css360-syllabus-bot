import { useCallback, useRef, useState } from 'react';
import type { ComparisonRun, ComparisonRunResponse } from '../context/comparisonRun';
import {
  generateBaseModel,
  generateFineTuned,
  generateFineTunedRag,
  generateRag,
} from '../lib/api';
import { toUserMessage } from '../lib/errorMessages';
import type { ModelKey } from '../types';
import { formatRagSourceLabels } from '../utils/ragSourceLabels';

export type ApproachState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; text: string; sources: string[] }
  | { status: 'error'; message: string };

export type ApproachStates = Record<ModelKey, ApproachState>;

const IDLE_STATES: ApproachStates = {
  base: { status: 'idle' },
  rag: { status: 'idle' },
  fineTuned: { status: 'idle' },
  fineTunedRag: { status: 'idle' },
};

function studentMessage(error: unknown): string {
  return toUserMessage(error, { audience: 'student', context: 'model-response' }).message;
}

function generateRunId(): string {
  return `run-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export interface UseComparisonRunResult {
  states: ApproachStates;
  activeQuestion: string;
  isRunning: boolean;
  run: (question: string, matchedComparisonId: string | null) => Promise<void>;
}

/**
 * Runs one question against all four approaches.
 *
 * The control flow here is carried over unchanged from the original compare
 * page and must stay that way:
 *
 *   - Base and RAG share one CPU-bound local model process, so they run
 *     strictly in sequence — RAG only starts once Base has settled.
 *   - The two fine-tuned paths hit a separate service and overlap with
 *     that chain.
 *   - A failure in one path never cancels another; each loader owns its own
 *     card state, and RAG still runs when Base fails.
 *   - `requestIdRef` discards results from a superseded run, and
 *     `isRunningRef` blocks a second submission while one is in flight.
 *
 * The only addition is `resultsRef`, which accumulates what each path produced
 * so the completed run can be handed to Evaluate.
 */
export function useComparisonRun(
  courseId: string,
  onComplete: (run: ComparisonRun) => void,
): UseComparisonRunResult {
  const [states, setStates] = useState<ApproachStates>(IDLE_STATES);
  const [activeQuestion, setActiveQuestion] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const requestIdRef = useRef(0);
  const isRunningRef = useRef(false);
  const resultsRef = useRef<Record<ModelKey, ComparisonRunResponse>>({
    base: { text: '', error: null, sources: [] },
    rag: { text: '', error: null, sources: [] },
    fineTuned: { text: '', error: null, sources: [] },
    fineTunedRag: { text: '', error: null, sources: [] },
  });

  const run = useCallback(
    async (question: string, matchedComparisonId: string | null) => {
      const trimmedQuestion = question.trim();

      if (!trimmedQuestion || isRunningRef.current) {
        return;
      }

      isRunningRef.current = true;
      setIsRunning(true);

      const requestId = ++requestIdRef.current;
      setActiveQuestion(trimmedQuestion);
      setStates({
        base: { status: 'loading' },
        rag: { status: 'loading' },
        fineTuned: { status: 'loading' },
        fineTunedRag: { status: 'loading' },
      });
      resultsRef.current = {
        base: { text: '', error: null, sources: [] },
        rag: { text: '', error: null, sources: [] },
        fineTuned: { text: '', error: null, sources: [] },
        fineTunedRag: { text: '', error: null, sources: [] },
      };

      const isCurrentRequest = () => requestId === requestIdRef.current;

      const succeed = (key: ModelKey, text: string, sources: string[] = []) => {
        if (!isCurrentRequest()) {
          return;
        }
        resultsRef.current[key] = { text, error: null, sources };
        setStates((current) => ({
          ...current,
          [key]: { status: 'success', text, sources },
        }));
      };

      const fail = (key: ModelKey, error: unknown) => {
        if (!isCurrentRequest()) {
          return;
        }
        const message = studentMessage(error);
        resultsRef.current[key] = { text: '', error: message, sources: [] };
        setStates((current) => ({
          ...current,
          [key]: { status: 'error', message },
        }));
      };

      const loadBase = async () => {
        try {
          const result = await generateBaseModel(courseId, trimmedQuestion);
          succeed('base', result.answer);
        } catch (error) {
          fail('base', error);
        }
      };

      const loadRag = async () => {
        try {
          const result = await generateRag(courseId, trimmedQuestion);
          succeed('rag', result.answer, formatRagSourceLabels(result.sources));
        } catch (error) {
          fail('rag', error);
        }
      };

      const loadFineTuned = async () => {
        try {
          const result = await generateFineTuned(courseId, trimmedQuestion);
          succeed('fineTuned', result.answer);
        } catch (error) {
          fail('fineTuned', error);
        }
      };

      const loadFineTunedRag = async () => {
        try {
          const result = await generateFineTunedRag(courseId, trimmedQuestion);
          succeed(
            'fineTunedRag',
            result.answer,
            formatRagSourceLabels(result.sources),
          );
        } catch (error) {
          fail('fineTunedRag', error);
        }
      };

      try {
        // Local model paths share one CPU-bound process. Run them sequentially
        // (Base, then RAG) to avoid contention. The fine-tuned paths use a
        // separate service and may overlap with the Base -> RAG chain. A Base
        // failure must not skip RAG; each loader isolates its own card state.
        await Promise.all([
          (async () => {
            await loadBase();
            await loadRag();
          })(),
          loadFineTuned(),
          loadFineTunedRag(),
        ]);
      } finally {
        if (isCurrentRequest()) {
          isRunningRef.current = false;
          setIsRunning(false);
          onComplete({
            runId: generateRunId(),
            courseId,
            question: trimmedQuestion,
            matchedComparisonId,
            createdAt: new Date().toISOString(),
            responses: { ...resultsRef.current },
          });
        }
      }
    },
    [courseId, onComplete],
  );

  return { states, activeQuestion, isRunning, run };
}
