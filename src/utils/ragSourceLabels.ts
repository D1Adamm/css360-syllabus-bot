import type { RagGenerateSource } from '../lib/api';

const GENERIC_SECTION_TITLES = new Set([
  'general',
  'introduction',
  'course introduction',
  'syllabus',
  'overview',
  'contents',
  'table of contents',
]);

const LEADING_FRAGMENT_PATTERN =
  /^(?:and|or|but|so|then|also|as|if|when|while|because|that|which|who|whom|whose|to|of|in|on|for|with|from|by|at|a|an|the|it|its|this|these|those|they|them|their)\b[\s,]*/i;

function normalizeTitle(title: string): string {
  return title.trim().toLowerCase().replace(/\s+/g, ' ');
}

export function isGenericSectionTitle(title: string): boolean {
  const cleaned = normalizeTitle(title);
  if (!cleaned) {
    return true;
  }
  if (GENERIC_SECTION_TITLES.has(cleaned)) {
    return true;
  }
  // Document-style titles such as "Software Engineering (Fall 2025)".
  if (/\b(19|20)\d{2}\b/.test(cleaned) && cleaned.split(' ').length <= 8) {
    return true;
  }
  return false;
}

function looksLikeHeadingLine(line: string): boolean {
  const stripped = line.trim();
  if (!stripped || stripped.length > 80) {
    return false;
  }
  if (stripped.endsWith('.') && !stripped.endsWith('...')) {
    return false;
  }
  const words = stripped.split(/\s+/).filter(Boolean);
  if (words.length === 0 || words.length > 12) {
    return false;
  }
  if (stripped.endsWith(':')) {
    return true;
  }
  if (/^\d+(?:\.\d+)*\s+\S/.test(stripped)) {
    return true;
  }
  if (stripped === stripped.toUpperCase() && words.length <= 8) {
    return true;
  }
  const titleCase = words.every((word) => /^[A-Z0-9]/.test(word));
  if (titleCase && words.length <= 8) {
    return true;
  }
  return false;
}

/** Strip leading conjunctions/punctuation so labels never start mid-sentence. */
export function cleanLabelFragment(text: string): string {
  let cleaned = text.replace(/\s+/g, ' ').trim();
  cleaned = cleaned.replace(/^[,;:\-–—.…]+/, '').trim();

  let previous = '';
  while (cleaned !== previous) {
    previous = cleaned;
    cleaned = cleaned.replace(LEADING_FRAGMENT_PATTERN, '').trim();
    cleaned = cleaned.replace(/^[,;:\-–—.…]+/, '').trim();
  }

  if (!cleaned) {
    return '';
  }
  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
}

function shortenPreview(text: string, maxLength = 72): string {
  const collapsed = cleanLabelFragment(text);
  if (!collapsed) {
    return '';
  }
  if (collapsed.length <= maxLength) {
    return collapsed;
  }
  const sliced = collapsed.slice(0, maxLength - 1);
  const boundary = sliced.lastIndexOf(' ');
  const preview = boundary > 40 ? sliced.slice(0, boundary) : sliced;
  return `${preview.replace(/[,;:\-–—]+$/, '')}…`;
}

function firstHeading(text: string, sectionTitle: string): string | null {
  for (const rawLine of text.split('\n')) {
    const line = rawLine.trim().replace(/:$/, '');
    if (!line) {
      continue;
    }
    if (normalizeTitle(line) === normalizeTitle(sectionTitle)) {
      continue;
    }
    if (looksLikeHeadingLine(line)) {
      return cleanLabelFragment(line);
    }
  }
  return null;
}

function firstCompleteSentence(text: string, sectionTitle: string): string | null {
  const withoutTitlePrefix = text
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line && normalizeTitle(line) !== normalizeTitle(sectionTitle))
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();

  if (!withoutTitlePrefix) {
    return null;
  }

  const match = withoutTitlePrefix.match(/^(.+?[.!?])(?:\s|$)/);
  const sentence = cleanLabelFragment(match?.[1] ?? withoutTitlePrefix);
  if (!sentence) {
    return null;
  }
  // Reject still-fragmentary leftovers that are too short to be useful.
  if (sentence.split(/\s+/).length < 4 && !sentence.endsWith('.')) {
    return null;
  }
  return shortenPreview(sentence, 72);
}

function distinguishingSubtitle(source: RagGenerateSource, title: string): string {
  const heading = firstHeading(source.text, title);
  if (heading && normalizeTitle(heading) !== normalizeTitle(title)) {
    return heading;
  }

  const sentence = firstCompleteSentence(source.text, title);
  if (sentence && normalizeTitle(sentence) !== normalizeTitle(title)) {
    return sentence;
  }

  return source.chunkId;
}

/**
 * Build readable syllabus source labels for the Compare page.
 *
 * Prefer, in order:
 * 1. meaningful sectionTitle
 * 2. detected heading from the chunk
 * 3. first complete sentence, cleaned and shortened
 * 4. chunk ID
 *
 * Generic or repeated titles always get a distinguishing subtitle.
 */
export function formatRagSourceLabels(sources: RagGenerateSource[]): string[] {
  const titleCounts = new Map<string, number>();
  for (const source of sources) {
    const title = source.sectionTitle.trim() || 'Syllabus excerpt';
    titleCounts.set(title, (titleCounts.get(title) ?? 0) + 1);
  }

  const labels = sources.map((source) => {
    const title = source.sectionTitle.trim() || 'Syllabus excerpt';
    const repeated = (titleCounts.get(title) ?? 0) > 1;
    const generic = isGenericSectionTitle(title);
    const subtitle = distinguishingSubtitle(source, title);

    if (!generic && !repeated) {
      return title;
    }

    if (generic) {
      return subtitle;
    }

    if (normalizeTitle(subtitle) === normalizeTitle(title)) {
      return `${title} (${source.chunkId})`;
    }
    return `${title}: ${subtitle}`;
  });

  // Final pass: disambiguate any remaining duplicate labels.
  const labelCounts = new Map<string, number>();
  for (const label of labels) {
    labelCounts.set(label, (labelCounts.get(label) ?? 0) + 1);
  }

  return labels.map((label, index) => {
    if ((labelCounts.get(label) ?? 0) <= 1) {
      return label;
    }
    const chunkId = sources[index]?.chunkId;
    return chunkId ? `${label} (${chunkId})` : label;
  });
}
