/**
 * Display formatting for course-identifying text.
 *
 * Course names are typed by whoever created the course, so the stored value is
 * whatever they entered — "Css 360", "css 360", "CSS360". A course code is an
 * abbreviation and reads wrong in title case, so it is normalised for display
 * only. The stored value is never modified.
 */

/** e.g. "CSS 360", "CSS360", "css-360a" — letters then digits, nothing else. */
const COURSE_CODE = /^\s*([A-Za-z]{2,6})\s*[-\s]?\s*(\d{2,4}[A-Za-z]?)\s*$/;

/**
 * Upper-cases the letters of a course code, and leaves anything else alone.
 *
 * Deliberately conservative: a name that is not code-shaped ("Intro to
 * Programming", "Senior Capstone") is returned untouched, because blanket
 * upper-casing would shout at every course whose name is a phrase.
 */
export function formatCourseCode(name: string | undefined | null): string {
  const value = (name ?? '').trim();
  if (!value) {
    return '';
  }

  const match = value.match(COURSE_CODE);
  if (!match) {
    return value;
  }

  const [, letters, digits] = match;
  return `${letters.toUpperCase()} ${digits.toUpperCase()}`;
}

/** Course name and title as one line, skipping whichever is missing. */
export function formatCourseHeading(
  name: string | undefined | null,
  title: string | undefined | null,
): string {
  const code = formatCourseCode(name);
  const subject = (title ?? '').trim();

  if (code && subject) {
    return `${code} — ${subject}`;
  }
  return code || subject || 'Course';
}
