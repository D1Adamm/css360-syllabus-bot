import type { Role } from '../context/role';
import { ApiError } from './api';

/**
 * Technical failures translated into language each audience can act on.
 *
 * Students and professors never see infrastructure. They are not the people
 * who can restart a service, set an environment variable, or read a stack
 * trace, so telling them to is noise at best and alarming at worst. Admin is
 * the only audience that gets the raw text, and even there it is secondary
 * detail rather than the headline.
 */

export type ErrorContext =
  | 'syllabus'
  | 'syllabus-upload'
  | 'course-create'
  | 'course-list'
  | 'examples-load'
  | 'example-save'
  | 'example-review'
  | 'model-response'
  | 'evaluation-save'
  | 'evaluation-load'
  | 'admin-operation';

export interface UserFacingError {
  title: string;
  message: string;
  /**
   * Raw technical text. Populated for admin only; every other audience gets
   * `undefined` so it cannot leak into a student or professor surface.
   */
  technical?: string;
}

/** Phrases that must never reach a student or professor surface. */
const TECHNICAL_MARKERS = [
  'fastapi',
  'uvicorn',
  'ollama',
  // The store's name is infrastructure to a student the same way the old one
  // was: backend 503s say "PostgreSQL is unavailable while …", and that is not
  // a sentence a professor should read on their own course page.
  'postgres',
  'database_url',
  'vite_',
  'finetuned_service_url',
  'env file',
  '.env',
  'localhost',
  '127.0.0.1',
  'tillicum',
  'slurm',
  'adapter',
  'rag index',
  'syllabus index',
  'traceback',
  'econnrefused',
];

/**
 * True when a backend `detail` is safe to show verbatim.
 *
 * Validation messages ("Only .pdf and .txt files are supported.") are written
 * for the person doing the action and are more useful than anything generic we
 * could substitute. Anything mentioning infrastructure is not.
 */
function isPresentableDetail(detail: string): boolean {
  if (!detail.trim()) {
    return false;
  }
  if (detail.includes('/') && /\/\w+\/\w+/.test(detail)) {
    // Looks like a filesystem or database path.
    return false;
  }
  const lower = detail.toLowerCase();
  return !TECHNICAL_MARKERS.some((marker) => lower.includes(marker));
}

function technicalText(error: unknown): string {
  if (error instanceof ApiError) {
    return error.status ? `${error.status}: ${error.message}` : error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

interface CopyEntry {
  title: string;
  student: string;
  professor: string;
}

const UNAVAILABLE: Record<ErrorContext, CopyEntry> = {
  syllabus: {
    title: 'Syllabus unavailable',
    student:
      'The syllabus could not be loaded right now. Try again in a moment.',
    professor:
      'The syllabus could not be loaded right now. Try again in a moment, or contact the project administrator if it continues.',
  },
  'syllabus-upload': {
    title: "We couldn't process this syllabus",
    student: 'The syllabus could not be processed. Try again in a moment.',
    professor:
      'Try uploading it again. If it keeps failing, contact the project administrator.',
  },
  'course-create': {
    title: "We couldn't create this course",
    student: 'Something went wrong. Try again in a moment.',
    professor:
      'Try again in a moment. If it keeps failing, contact the project administrator.',
  },
  'course-list': {
    title: 'Courses unavailable',
    student: 'Your courses could not be loaded. Try again in a moment.',
    professor: 'Your courses could not be loaded. Try again in a moment.',
  },
  'examples-load': {
    title: 'Examples unavailable',
    student: 'Your questions could not be loaded. Try again in a moment.',
    professor: 'These examples could not be loaded. Try again in a moment.',
  },
  'example-save': {
    title: "We couldn't save this",
    student: 'Your question was not saved. Try again in a moment.',
    professor: 'This change was not saved. Try again in a moment.',
  },
  'example-review': {
    title: "We couldn't save your review",
    student: 'This change was not saved. Try again in a moment.',
    professor: 'Your review was not saved. Try again in a moment.',
  },
  'model-response': {
    title: 'Response unavailable',
    student: 'This response is temporarily unavailable. Try again in a moment.',
    professor: 'This response is temporarily unavailable. Try again in a moment.',
  },
  'evaluation-save': {
    title: "We couldn't save your evaluation",
    student: 'Your evaluation was not saved. Try again in a moment.',
    professor: 'This evaluation was not saved. Try again in a moment.',
  },
  'evaluation-load': {
    title: 'Results unavailable',
    student: 'Results could not be loaded. Try again in a moment.',
    professor: 'Results could not be loaded. Try again in a moment.',
  },
  'admin-operation': {
    title: 'Operation failed',
    student: 'Something went wrong. Try again in a moment.',
    professor: 'Something went wrong. Try again in a moment.',
  },
};

/** `app.generation_queue.BUSY_CODE`. */
export const GENERATION_BUSY_CODE = 'generation_busy';

/**
 * `app.error_classification.EVALUATION_ALREADY_RECORDED`: the backend already
 * holds this participant's rating of this comparison run. Not a failure the
 * student can retry — the Evaluate page treats it as "rated".
 */
export const EVALUATION_ALREADY_RECORDED_CODE = 'evaluation_already_recorded';

/**
 * The backend's classified failures (`app.error_classification`), in words for
 * the person who asked.
 *
 * Until 2026-11-03 every refusal from the fine-tuned path arrived as a bare
 * 503 and every one of them read "temporarily unavailable" (audit findings B5
 * and B7). A course with no activated model, a service that was down, an
 * answer that timed out and a syllabus that was never indexed all looked the
 * same to a student and to the professor trying to help them. The backend now
 * sends a `code` with each; this table says what each one means and what, if
 * anything, the reader can do. Still no infrastructure: the administrator's
 * Overview page carries the detail, and admin keeps the raw message.
 */
const CLASSIFIED_COPY: Record<string, CopyEntry> = {
  model_not_activated: {
    title: 'No published course model yet',
    student:
      'This course does not have a published course model yet, so the Fine-Tuned answers cannot be generated. The Base and RAG answers still work.',
    professor:
      'Your course model has not been published yet, so the Fine-Tuned answers cannot be generated. The Base and RAG answers still work; the Model page shows where things stand.',
  },
  model_version_not_servable: {
    title: 'That model version is not available',
    student:
      'The course model version requested is not available right now. The Base and RAG answers still work.',
    professor:
      'That version of the course model is not published or not loaded right now. The Base and RAG answers still work.',
  },
  finetuned_not_configured: {
    title: 'Course model answers unavailable',
    student:
      'Fine-Tuned answers are not available right now. The Base and RAG answers still work.',
    professor:
      'Fine-Tuned answers are not available right now; the course model service is not set up. The Base and RAG answers still work. Contact the project administrator.',
  },
  finetuned_unavailable: {
    title: 'Course model answers unavailable',
    student:
      'The course model is not answering right now. The Base and RAG answers still work; try again in a few minutes.',
    professor:
      'The course model service is not answering right now. The Base and RAG answers still work. If it continues, contact the project administrator.',
  },
  finetuned_upstream_error: {
    title: 'Course model answers unavailable',
    student:
      'The course model could not answer this time. The Base and RAG answers still work; try again in a moment.',
    professor:
      'The course model service reported a problem answering this. The Base and RAG answers still work. If it continues, contact the project administrator.',
  },
  finetuned_bad_response: {
    title: 'Course model answers unavailable',
    student:
      'The course model could not answer this time. The Base and RAG answers still work; try again in a moment.',
    professor:
      'The course model returned an answer that could not be used. The Base and RAG answers still work. If it continues, contact the project administrator.',
  },
  finetuned_rejected: {
    title: 'Course model answers unavailable',
    student:
      'The course model could not take this question. The Base and RAG answers still work; try rephrasing it.',
    professor:
      'The course model service refused this request. The Base and RAG answers still work. If it continues, contact the project administrator.',
  },
  finetuned_timeout: {
    title: 'This answer took too long',
    student:
      'The course model did not finish answering in time. Try again in a moment; shorter questions usually finish faster.',
    professor:
      'The course model did not finish answering in time. Try again in a moment; if it keeps happening, the project administrator can check the service.',
  },
  ollama_timeout: {
    title: 'This answer took too long',
    student:
      'The model did not finish answering in time. Try again in a moment; shorter questions usually finish faster.',
    professor:
      'The model did not finish answering in time. Try again in a moment; if it keeps happening, the project administrator can check the service.',
  },
  ollama_unavailable: {
    title: 'Response unavailable',
    student:
      'The model that generates this answer is not responding right now. Try again in a few minutes.',
    professor:
      'The model that generates this answer is not responding right now. Try again in a few minutes; if it continues, contact the project administrator.',
  },
  ollama_upstream_error: {
    title: 'Response unavailable',
    student: 'The model could not answer this time. Try again in a moment.',
    professor:
      'The model reported a problem answering this. Try again in a moment; if it continues, contact the project administrator.',
  },
  ollama_bad_response: {
    title: 'Response unavailable',
    student: 'The model could not answer this time. Try again in a moment.',
    professor:
      'The model returned an answer that could not be used. Try again in a moment; if it continues, contact the project administrator.',
  },
  syllabus_not_indexed: {
    title: 'No syllabus to search yet',
    student:
      "This course's syllabus has not been prepared yet, so there is nothing to search. The Base answer does not need it.",
    professor:
      'The syllabus has not been prepared for searching yet. Upload or re-process it on the Syllabus page; until then the RAG answers have nothing to search.',
  },
  [EVALUATION_ALREADY_RECORDED_CODE]: {
    title: 'Already rated',
    student: 'You have already rated this comparison.',
    professor: 'This comparison has already been rated.',
  },
};

/** Whether `toUserMessage` has dedicated copy for a backend code. */
export function isClassifiedCode(code: string | undefined): boolean {
  return typeof code === 'string' && code in CLASSIFIED_COPY;
}

export const GENERATION_BUSY_MESSAGE =
  'Many people are asking at the same moment and this answer could not start in time. Please try again in a minute.';

export interface ToUserMessageOptions {
  audience: Role;
  context: ErrorContext;
}

export function toUserMessage(
  error: unknown,
  { audience, context }: ToUserMessageOptions,
): UserFacingError {
  const copy = UNAVAILABLE[context];
  const raw = technicalText(error);

  if (audience === 'admin') {
    return {
      title: copy.title,
      message: error instanceof ApiError ? error.message : copy.professor,
      technical: raw,
    };
  }

  const fallback = audience === 'professor' ? copy.professor : copy.student;

  // The generation queue's capacity refusal. A 503, but not an outage: the
  // backend is up and saying "too many at once", in words meant for the
  // person who asked, so they know waiting a minute is the fix.
  if (error instanceof ApiError && error.code === GENERATION_BUSY_CODE) {
    return { title: 'Busy right now', message: GENERATION_BUSY_MESSAGE };
  }

  // A classified refusal: the backend said which kind of problem this is, so
  // the reader gets that rather than the one sentence every outage used to
  // share. Each entry names what still works and what, if anything, to do.
  if (error instanceof ApiError && error.code && CLASSIFIED_COPY[error.code]) {
    const classified = CLASSIFIED_COPY[error.code]!;
    return {
      title: classified.title,
      message: audience === 'professor' ? classified.professor : classified.student,
    };
  }

  // A 4xx usually carries a message written for the person doing the action.
  // Anything else is an infrastructure problem they cannot act on.
  if (
    error instanceof ApiError &&
    typeof error.status === 'number' &&
    error.status >= 400 &&
    error.status < 500 &&
    isPresentableDetail(error.message)
  ) {
    return { title: copy.title, message: error.message };
  }

  return { title: copy.title, message: fallback };
}

/** Convenience for the many places that only render a single line. */
export function toUserMessageText(
  error: unknown,
  options: ToUserMessageOptions,
): string {
  return toUserMessage(error, options).message;
}
