import { Fragment, useEffect, useState } from 'react';
import { PageHeader } from '../components/PageHeader';
import { useCourseId } from '../context/CourseContext';
import { ApiError, fetchCourseSyllabusText } from '../lib/api';
import { parseSyllabusDocument } from '../lib/syllabusDocument';

type SyllabusPageState =
  | { status: 'loading' }
  | { status: 'success'; text: string; characterCount: number }
  | { status: 'empty' }
  | { status: 'not-found' }
  | { status: 'invalid-course'; message: string }
  | { status: 'unavailable'; message: string }
  | { status: 'error'; message: string };

function getErrorState(error: unknown): Exclude<SyllabusPageState, { status: 'loading' | 'success' | 'empty' }> {
  if (error instanceof ApiError) {
    if (error.status === 404) {
      return { status: 'not-found' };
    }

    if (error.status === 400) {
      return {
        status: 'invalid-course',
        message: error.message || 'This course id is not valid.',
      };
    }

    if (error.status === undefined) {
      return {
        status: 'unavailable',
        message:
          error.message ||
          'Could not reach the backend to load the syllabus. Make sure the FastAPI server is running.',
      };
    }

    return {
      status: 'error',
      message: error.message || 'The syllabus could not be loaded for this course.',
    };
  }

  return {
    status: 'unavailable',
    message: 'Could not reach the backend to load the syllabus. Make sure the FastAPI server is running.',
  };
}

export function SyllabusPage() {
  const courseId = useCourseId();
  const [state, setState] = useState<SyllabusPageState>({ status: 'loading' });

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
        if (!cancelled) {
          setState(getErrorState(error));
        }
      });

    return () => {
      cancelled = true;
    };
  }, [courseId]);

  return (
    <>
      <PageHeader
        title="Syllabus"
        description={`Extracted syllabus text for course ${courseId}. This is the text used to build the course-specific RAG index.`}
      />

      <aside className="syllabus-notice" aria-label="Syllabus source note">
        <p>
          <strong>Course syllabus:</strong> Content is loaded from the extracted{' '}
          <code>syllabus.txt</code> for <code>{courseId}</code>. Always consult the official
          syllabus on Canvas and weekly announcements for the most current details.
        </p>
      </aside>

      {state.status === 'loading' ? (
        <section className="syllabus-state" aria-live="polite" aria-busy="true">
          <h2 className="syllabus-state__title">Loading syllabus</h2>
          <p className="syllabus-state__text">
            Fetching the extracted syllabus text for this course…
          </p>
        </section>
      ) : null}

      {state.status === 'not-found' ? (
        <section className="syllabus-state" aria-live="polite">
          <h2 className="syllabus-state__title">Syllabus not found</h2>
          <p className="syllabus-state__text">
            No extracted syllabus text is available for <code>{courseId}</code> yet. Upload a
            syllabus for this course to generate <code>syllabus.txt</code>.
          </p>
        </section>
      ) : null}

      {state.status === 'empty' ? (
        <section className="syllabus-state" aria-live="polite">
          <h2 className="syllabus-state__title">Empty syllabus</h2>
          <p className="syllabus-state__text">
            The extracted syllabus file for <code>{courseId}</code> exists but does not contain
            displayable text.
          </p>
        </section>
      ) : null}

      {state.status === 'invalid-course' ? (
        <section className="syllabus-state" aria-live="polite">
          <h2 className="syllabus-state__title">Invalid course</h2>
          <p className="syllabus-state__text">{state.message}</p>
        </section>
      ) : null}

      {state.status === 'unavailable' ? (
        <section className="syllabus-state" aria-live="polite">
          <h2 className="syllabus-state__title">Backend unavailable</h2>
          <p className="syllabus-state__text">{state.message}</p>
        </section>
      ) : null}

      {state.status === 'error' ? (
        <section className="syllabus-state" aria-live="polite">
          <h2 className="syllabus-state__title">Unable to load syllabus</h2>
          <p className="syllabus-state__text">{state.message}</p>
        </section>
      ) : null}

      {state.status === 'success' ? (
        <section
          className="syllabus-document-section"
          aria-label="Extracted syllabus text"
          aria-live="polite"
        >
          <p className="syllabus-document-meta">
            {state.characterCount.toLocaleString()} characters
          </p>
          <article className="syllabus-document" data-testid="syllabus-document">
            {parseSyllabusDocument(state.text).map((block, index) => {
              if (block.type === 'heading') {
                return (
                  <h2 key={`heading-${index}`} className="syllabus-document__heading">
                    {block.text}
                  </h2>
                );
              }

              return (
                <p key={`paragraph-${index}`} className="syllabus-document__paragraph">
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
        </section>
      ) : null}
    </>
  );
}
