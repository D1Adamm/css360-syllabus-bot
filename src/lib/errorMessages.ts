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
