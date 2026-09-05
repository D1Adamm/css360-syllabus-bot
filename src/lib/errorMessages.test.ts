import { describe, expect, it } from 'vitest';
import { ApiError } from './api';
import { toUserMessage, type ErrorContext } from './errorMessages';

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

  it('describes a non-Error value without throwing', () => {
    const result = toUserMessage('something odd', {
      audience: 'admin',
      context: 'admin-operation',
    });
    expect(result.technical).toBe('something odd');
  });
});
