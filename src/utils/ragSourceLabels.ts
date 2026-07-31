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

function firstMeaningfulLine(text: string, sectionTitle: string): string | null {
  for (const rawLine of text.split('\n')) {
    const line = rawLine.trim();
    if (!line) {
      continue;
    }
    if (normalizeTitle(line) === normalizeTitle(sectionTitle)) {
      continue;
    }
    return line.replace(/:$/, '');
  }
  return null;
}

function shortenPreview(text: string, maxLength = 72): string {
  const collapsed = text.replace(/\s+/g, ' ').trim();
  if (collapsed.length <= maxLength) {
    return collapsed;
  }
  const sliced = collapsed.slice(0, maxLength - 1);
  const boundary = sliced.lastIndexOf(' ');
  const preview = boundary > 40 ? sliced.slice(0, boundary) : sliced;
  return `${preview}…`;
}

/**
 * Build readable syllabus source labels for the Compare page.
 * Prefer subsection headings; when titles are generic or repeated, add a
 * short text preview or chunk id so entries stay distinguishable.
 */
export function formatRagSourceLabels(sources: RagGenerateSource[]): string[] {
  const titleCounts = new Map<string, number>();
  for (const source of sources) {
    const title = source.sectionTitle.trim() || 'Syllabus excerpt';
    titleCounts.set(title, (titleCounts.get(title) ?? 0) + 1);
  }

  return sources.map((source) => {
    const title = source.sectionTitle.trim() || 'Syllabus excerpt';
    const repeated = (titleCounts.get(title) ?? 0) > 1;
    const generic = isGenericSectionTitle(title);
    const heading = firstMeaningfulLine(source.text, title);
    const headingIsUseful =
      Boolean(heading) &&
      looksLikeHeadingLine(heading!) &&
      normalizeTitle(heading!) !== normalizeTitle(title);

    if (!generic && !repeated) {
      return title;
    }

    if (headingIsUseful && heading) {
      return generic ? heading : `${title}: ${heading}`;
    }

    const preview = shortenPreview(source.text);
    if (generic) {
      return preview || source.chunkId;
    }
    if (repeated && preview) {
      return `${title} — ${preview}`;
    }
    return `${title} (${source.chunkId})`;
  });
}
