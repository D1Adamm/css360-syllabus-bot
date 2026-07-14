export type SyllabusDocumentBlock =
  | { type: 'heading'; text: string }
  | { type: 'paragraph'; lines: string[] };

/**
 * Split extracted syllabus text into heading/paragraph blocks so the UI can
 * preserve spacing and line breaks instead of collapsing into one paragraph.
 */
export function parseSyllabusDocument(text: string): SyllabusDocumentBlock[] {
  const normalized = text.replace(/\r\n/g, '\n').trim();
  if (normalized === '') {
    return [];
  }

  return normalized
    .split(/\n{2,}/)
    .map((block) => block.trimEnd())
    .filter((block) => block.trim() !== '')
    .map((block) => {
      const lines = block.split('\n');
      if (isHeadingBlock(lines)) {
        return { type: 'heading', text: lines[0]!.trim() };
      }

      return { type: 'paragraph', lines };
    });
}

function isHeadingBlock(lines: string[]): boolean {
  if (lines.length !== 1) {
    return false;
  }

  const line = lines[0]!.trim();
  if (line.length === 0 || line.length > 90) {
    return false;
  }

  if (/^[•\-*]/.test(line) || /^\d+[.)]\s+\S/.test(line)) {
    return false;
  }

  if (/[.!?:]$/.test(line)) {
    return false;
  }

  return true;
}
