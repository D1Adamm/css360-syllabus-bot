/**
 * What the backend accepts as a syllabus file, checked before uploading so a
 * professor gets a clear message instead of a refused request. The backend
 * checks again (`backend/app/syllabus_upload.py`); these must agree with it.
 */

export const MAX_SYLLABUS_BYTES = 10 * 1024 * 1024;

const ALLOWED_SYLLABUS_EXTENSIONS = new Set(['pdf', 'txt']);

function getFileExtension(fileName: string): string {
  const parts = fileName.toLowerCase().split('.');
  return parts.length > 1 ? (parts.at(-1) ?? '') : '';
}

/** A message to show, or undefined when the file can be uploaded. */
export function validateSyllabusFile(file: File | null): string | undefined {
  if (!file) {
    return 'A PDF or TXT syllabus file is required.';
  }
  if (!ALLOWED_SYLLABUS_EXTENSIONS.has(getFileExtension(file.name))) {
    return 'Only .pdf and .txt syllabus files are supported.';
  }
  if (file.size > MAX_SYLLABUS_BYTES) {
    return 'Syllabus files must be 10 MB or smaller.';
  }
  return undefined;
}
