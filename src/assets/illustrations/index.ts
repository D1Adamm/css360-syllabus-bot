/**
 * Illustration registry.
 *
 * Every slot below is a planned, named place in the UI where a custom
 * illustration may appear. All slots currently resolve to `null`, and the
 * `Illustration` component renders a token-coloured geometric fallback in that
 * case — the layout is identical whether or not an asset exists, so artwork can
 * be added later without touching any page.
 *
 * To wire up a real asset:
 *   1. Drop the file in this directory, e.g. `landing.svg`.
 *   2. `import landing from './landing.svg';`
 *   3. Replace the `null` below with `landing`.
 *
 * Prefer SVG. Keep artwork within the brand palette in `styles/tokens.css`,
 * and avoid generic "AI sparkle" imagery or stock photography.
 */

export type IllustrationName =
  /** Public / landing experience: education + syllabus themed. */
  | 'landing'
  /** A course with no syllabus or no content yet: document themed. */
  | 'empty-course'
  /** Student contribution intro / empty state: document -> question. */
  | 'contribute'
  /** Course model finished preparing: small, quiet celebration. */
  | 'model-ready';

export const ILLUSTRATION_NAMES: IllustrationName[] = [
  'landing',
  'empty-course',
  'contribute',
  'model-ready',
];

/**
 * Source URL for each slot, or `null` when no asset has been provided yet.
 * Vite resolves an imported `.svg` to its emitted URL.
 */
export const ILLUSTRATION_SOURCES: Record<IllustrationName, string | null> = {
  landing: null,
  'empty-course': null,
  contribute: null,
  'model-ready': null,
};

/** Default alt text per slot, used when an illustration is not decorative. */
export const ILLUSTRATION_ALT: Record<IllustrationName, string> = {
  landing: 'Students and an assistant working from a course syllabus',
  'empty-course': 'An empty course syllabus document',
  contribute: 'A syllabus document turning into a question',
  'model-ready': 'A prepared course model ready to use',
};

export function getIllustrationSource(name: IllustrationName): string | null {
  return ILLUSTRATION_SOURCES[name];
}
