import { Fragment, useEffect, useState } from 'react';
import { ErrorState } from '../ui/ErrorState';
import { EmptyState } from '../ui/EmptyState';
import { PageHeader } from '../ui/PageHeader';
import { useCourseId } from '../../context/CourseContext';
import type { Role } from '../../context/role';
import { ApiError, fetchCourseSyllabusText } from '../../lib/api';
import { toUserMessage } from '../../lib/errorMessages';
import { parseSyllabusDocument } from '../../lib/syllabusDocument';

type SyllabusViewState =
  | { status: 'loading' }
  | { status: 'success'; text: string; characterCount: number }
  | { status: 'empty' }
  | { status: 'missing' }
  | { status: 'error'; message: string };

export interface SyllabusViewProps {
  audience: Role;
}

/**
 * Reads the course syllabus. Shared by the student and professor routes.
 *
 * The error branches map exactly onto the ones the page has always had — 404,
 * 400, unreachable, other — only the wording changed. Classification is
 * unchanged so behaviour is identical.
 */
export function SyllabusView({ audience }: SyllabusViewProps) {
  const courseId = useCourseId();
  const [state, setState] = useState<SyllabusViewState>({ status: 'loading' });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setState({ status: 'loading' });

    void fetchCourseSyllabusText(courseId)
      .then((result) => {
        if (cancelled) {
          return;
        }
        if (result.text.trim() === '') {
          setState({ status: 'empty' });
          return;
        }
        setState({
          status: 'success',
          text: result.text,
          characterCount: result.characterCount,
        });
      })
      .catch((error: unknown) => {
        if (cancelled) {
          return;
        }
        // 404: no syllabus added yet. 400: the course id in the URL is not
        // usable. Neither is a service failure, and neither should surface the
        // backend's internal wording about id formats.
        if (
          error instanceof ApiError &&
          (error.status === 404 || error.status === 400)
        ) {
          setState({ status: 'missing' });
          return;
        }
        setState({
          status: 'error',
          message: toUserMessage(error, { audience, context: 'syllabus' }).message,
        });
      });

    return () => {
      cancelled = true;
    };
  }, [courseId, audience, attempt]);

  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        title="Syllabus"
        description={
          audience === 'professor'
            ? 'The syllabus text your students see, and the text the course assistant draws on.'
            : "The course syllabus as your instructor provided it. For the latest updates, check your instructor's official course channels."
        }
      />

      {state.status === 'loading' && (
        <p className="ui-text-muted" role="status" aria-live="polite">
          Loading the syllabus…
        </p>
      )}

      {state.status === 'missing' && (
        <EmptyState
          illustration="empty-course"
          title="No syllabus yet"
          description={
            audience === 'professor'
              ? 'Upload a syllabus for this course and it will appear here.'
              : 'Your instructor has not added a syllabus for this course yet.'
          }
        />
      )}

      {state.status === 'empty' && (
        <EmptyState
          illustration="empty-course"
          title="Nothing to show"
          description="The syllabus for this course does not contain any readable text."
        />
      )}

      {state.status === 'error' && (
        <ErrorState
          title="Syllabus unavailable"
          message={state.message}
          onRetry={() => setAttempt((current) => current + 1)}
        />
      )}

      {state.status === 'success' && (
        <SyllabusDocument text={state.text} />
      )}
    </div>
  );
}

function headingSlug(text: string, index: number): string {
  const slug = text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return `syllabus-${slug || 'section'}-${index}`;
}

/**
 * The syllabus as something to read.
 *
 * Set in a single measured column rather than a bordered panel, with the
 * section list alongside it — a syllabus is long, and "where is the late
 * policy" is the question students actually arrive with. The contents list
 * only appears when there are enough sections to be worth scanning.
 */
function SyllabusDocument({ text }: { text: string }) {
  const blocks = parseSyllabusDocument(text);

  const headings = blocks
    .map((block, index) =>
      block.type === 'heading'
        ? { id: headingSlug(block.text, index), text: block.text }
        : null,
    )
    .filter((heading): heading is { id: string; text: string } => heading !== null);

  const hasContents = headings.length >= 3;

  return (
    <div
      className={hasContents ? 'syllabus syllabus--with-contents' : 'syllabus'}
      aria-live="polite"
    >
      {hasContents && (
        <nav className="syllabus__contents" aria-label="Syllabus sections">
          <p className="syllabus__contents-title">Sections</p>
          <ul className="syllabus__contents-list">
            {headings.map((heading) => (
              <li key={heading.id}>
                <a href={`#${heading.id}`}>{heading.text}</a>
              </li>
            ))}
          </ul>
        </nav>
      )}

      <article className="syllabus__body" data-testid="syllabus-document">
        {blocks.map((block, index) => {
          if (block.type === 'heading') {
            return (
              <h2
                key={`heading-${index}`}
                id={headingSlug(block.text, index)}
                className="syllabus__heading"
              >
                {block.text}
              </h2>
            );
          }

          return (
            <p key={`paragraph-${index}`} className="syllabus__paragraph">
              {block.lines.map((line, lineIndex) => (
                <Fragment key={`line-${index}-${lineIndex}`}>
                  {line}
                  {lineIndex < block.lines.length - 1 ? <br /> : null}
                </Fragment>
              ))}
            </p>
          );
        })}
      </article>
    </div>
  );
}
