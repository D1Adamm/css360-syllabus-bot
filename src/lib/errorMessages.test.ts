import { describe, expect, it } from 'vitest';
import { ApiError } from './api';
import { isClassifiedCode, toUserMessage, type ErrorContext } from './errorMessages';

const ALL_CONTEXTS: ErrorContext[] = [
  'syllabus',
  'syllabus-upload',
  'course-create',
  'course-list',
  'examples-load',
  'example-save',
  'example-review',
  'model-response',
  'evaluation-save',
  'evaluation-load',
  'admin-operation',
];

/**
 * The infrastructure vocabulary that must never reach a student or professor.
 * This is the regression guard for the whole redesign: if someone later routes
 * a raw backend string into a user-facing surface through this helper, one of
 * these cases fails.
 */
const FORBIDDEN =
  /fastapi|uvicorn|postgres|database_url|ollama|vite_|finetuned_service_url|\.env|localhost|127\.0\.0\.1|tillicum|slurm|traceback/i;

const LEAKY_ERRORS: unknown[] = [
  new ApiError('Could not reach the backend. Make sure the FastAPI server is running.'),
  new ApiError('API base URL is not configured. Set VITE_API_BASE_URL in your .env file.'),
  new ApiError('FINETUNED_SERVICE_URL is not set.', 500),
  new ApiError('Ollama returned an error', 502),
  new Error('PostgreSQL permission denied for relation seed_examples'),
  new Error('connect ECONNREFUSED 127.0.0.1:8001'),
  'Slurm job failed on tillicum',
];

describe('toUserMessage', () => {
  it('never leaks infrastructure wording to students or professors', () => {
    for (const context of ALL_CONTEXTS) {
      for (const audience of ['student', 'professor'] as const) {
        for (const error of LEAKY_ERRORS) {
          const result = toUserMessage(error, { audience, context });
          expect(result.message).not.toMatch(FORBIDDEN);
          expect(result.title).not.toMatch(FORBIDDEN);
          // Raw technical text is admin-only.
          expect(result.technical).toBeUndefined();
        }
      }
    }
  });

  it('always produces a non-empty title and message', () => {
    for (const context of ALL_CONTEXTS) {
      const result = toUserMessage(new Error('boom'), { audience: 'student', context });
      expect(result.title.length).toBeGreaterThan(0);
      expect(result.message.length).toBeGreaterThan(0);
    }
  });

  it('passes through a 4xx validation message that was written for the user', () => {
    const result = toUserMessage(
      new ApiError('Only .pdf and .txt syllabus files are supported.', 400),
      { audience: 'professor', context: 'syllabus-upload' },
    );
    expect(result.message).toBe('Only .pdf and .txt syllabus files are supported.');
  });

  it('suppresses a 4xx message that names infrastructure', () => {
    const result = toUserMessage(
      new ApiError('PostgreSQL rejected the write', 400),
      { audience: 'professor', context: 'example-save' },
    );
    expect(result.message).not.toMatch(/postgres/i);
  });

  it('suppresses a 4xx message that carries a filesystem path', () => {
    const result = toUserMessage(
      new ApiError('Missing backend/data/indexes/course.json', 404),
      { audience: 'student', context: 'syllabus' },
    );
    expect(result.message).not.toMatch(/backend\/data/);
  });

  it('shows a student the neutral no-model refusal exactly as the backend wrote it', () => {
    // The 409 a course without a fine-tuned model returns. It is written for
    // the student, so it must pass through unchanged rather than be replaced
    // by the generic "temporarily unavailable" line, which would suggest a
    // retry could help.
    const result = toUserMessage(
      new ApiError('A fine-tuned model is not available for this course yet.', 409),
      { audience: 'student', context: 'model-response' },
    );
    expect(result.message).toBe(
      'A fine-tuned model is not available for this course yet.',
    );
    expect(result.message).not.toMatch(/train one|before asking/i);
  });

  it('does not pass a 5xx message through, even when it looks harmless', () => {
    const result = toUserMessage(new ApiError('Internal error', 500), {
      audience: 'student',
      context: 'model-response',
    });
    expect(result.message).toBe(
      'This response is temporarily unavailable. Try again in a moment.',
    );
  });

  it('gives admin the raw technical detail', () => {
    const result = toUserMessage(new ApiError('Ollama refused the connection', 502), {
      audience: 'admin',
      context: 'admin-operation',
    });
    expect(result.technical).toContain('502');
    expect(result.technical).toContain('Ollama refused the connection');
  });

  it('tells a student plainly that the models are busy, rather than "unavailable"', () => {
    const busy = new ApiError('anything', 503, 'generation_busy');
    for (const audience of ['student', 'professor'] as const) {
      const result = toUserMessage(busy, { audience, context: 'model-response' });
      expect(result.title).toBe('Busy right now');
      expect(result.message).toMatch(/try again in a minute/i);
    }
  });

  it('describes a non-Error value without throwing', () => {
    const result = toUserMessage('something odd', {
      audience: 'admin',
      context: 'admin-operation',
    });
    expect(result.technical).toBe('something odd');
  });

  /* --------------------------------------------------------------------- *
   * Classified failures (audit findings B5/B7, 2026-11-03)
   *
   * The backend now says which kind of problem a refusal is. Each kind gets
   * its own sentence, so "no model has been published for this course" no
   * longer reads the same as "the service is down", and the reader is told
   * what still works.
   * --------------------------------------------------------------------- */

  const CLASSIFIED_CODES = [
    'model_not_activated',
    'model_version_not_servable',
    'finetuned_not_configured',
    'finetuned_unavailable',
    'finetuned_upstream_error',
    'finetuned_bad_response',
    'finetuned_rejected',
    'finetuned_timeout',
    'ollama_timeout',
    'ollama_unavailable',
    'ollama_upstream_error',
    'ollama_bad_response',
    'syllabus_not_indexed',
    'evaluation_already_recorded',
  ];

  it('has copy for every classified code the backend sends', () => {
    for (const code of CLASSIFIED_CODES) {
      expect(isClassifiedCode(code)).toBe(true);
    }
    expect(isClassifiedCode('something_else')).toBe(false);
    expect(isClassifiedCode(undefined)).toBe(false);
  });

  it('distinguishes a course with no published model from a service outage', () => {
    const notActivated = toUserMessage(
      new ApiError('Fine-tuned answers are not available for this course yet.', 503, 'model_not_activated'),
      { audience: 'student', context: 'model-response' },
    );
    const down = toUserMessage(
      new ApiError('The fine-tuned inference service is unavailable.', 503, 'finetuned_unavailable'),
      { audience: 'student', context: 'model-response' },
    );
    const slow = toUserMessage(
      new ApiError('The fine-tuned inference service timed out.', 504, 'finetuned_timeout'),
      { audience: 'student', context: 'model-response' },
    );

    expect(notActivated.message).toMatch(/does not have a published course model/);
    expect(down.message).toMatch(/not answering right now/);
    expect(slow.message).toMatch(/did not finish answering in time/);
    expect(new Set([notActivated.message, down.message, slow.message]).size).toBe(3);
    // Each says what still works.
    expect(notActivated.message).toMatch(/Base and RAG answers still work/);
    expect(down.message).toMatch(/Base and RAG answers still work/);
  });

  it('points a professor at what they can do, and a student at what still works', () => {
    const error = new ApiError('No usable syllabus context.', 404, 'syllabus_not_indexed');
    expect(
      toUserMessage(error, { audience: 'professor', context: 'model-response' }).message,
    ).toMatch(/Syllabus page/);
    expect(
      toUserMessage(error, { audience: 'student', context: 'model-response' }).message,
    ).toMatch(/has not been prepared yet/);
  });

  it('keeps every classified sentence free of infrastructure wording', () => {
    for (const code of CLASSIFIED_CODES) {
      const error = new ApiError('Ollama at 127.0.0.1:11435 refused (see FINETUNED_SERVICE_URL in .env)', 503, code);
      for (const audience of ['student', 'professor'] as const) {
        const result = toUserMessage(error, { audience, context: 'model-response' });
        expect(result.message).not.toMatch(FORBIDDEN);
        expect(result.title).not.toMatch(FORBIDDEN);
        expect(result.message).not.toMatch(/adapter|127\.0\.0\.1/i);
        expect(result.technical).toBeUndefined();
      }
    }
  });

  it('still gives admin the raw classified message', () => {
    const result = toUserMessage(
      new ApiError('The fine-tuned inference service is unavailable.', 503, 'finetuned_unavailable'),
      { audience: 'admin', context: 'model-response' },
    );
    expect(result.message).toBe('The fine-tuned inference service is unavailable.');
    expect(result.technical).toContain('503');
  });
});
