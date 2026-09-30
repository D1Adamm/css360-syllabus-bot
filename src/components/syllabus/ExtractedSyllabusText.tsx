import { useCallback, useEffect, useId, useRef, useState } from 'react';
import { Button } from '../ui/Button';
import { ApiError, fetchCourseSyllabusText } from '../../lib/api';
import { toUserMessage } from '../../lib/errorMessages';

type TextState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'ready'; text: string; characterCount: number }
  | { status: 'missing' }
  | { status: 'error'; message: string };

export interface ExtractedSyllabusTextProps {
  courseId: string;
  /** Change it to read the text again, e.g. after a new syllabus was uploaded. */
  refreshToken?: number;
}

/**
 * The text the course assistant actually indexed, for checking what was
 * parsed. Closed by default and fetched only when opened; shown verbatim in a
 * bounded box that scrolls on its own, so a long syllabus never makes the page
 * long.
 */
export function ExtractedSyllabusText({ courseId, refreshToken = 0 }: ExtractedSyllabusTextProps) {
  const regionId = useId();
  const [open, setOpen] = useState(false);
  const [state, setState] = useState<TextState>({ status: 'idle' });
  // Only the latest request may write its result: a new syllabus or course
  // makes any request still in flight stale.
  const latestRequest = useRef(0);

  const load = useCallback(() => {
    const request = ++latestRequest.current;
    setState({ status: 'loading' });
    fetchCourseSyllabusText(courseId)
      .then((result) => {
        if (latestRequest.current === request) {
          setState({ status: 'ready', text: result.text, characterCount: result.characterCount });
        }
      })
      .catch((error: unknown) => {
        if (latestRequest.current !== request) {
          return;
        }
        if (error instanceof ApiError && error.status === 404) {
          setState({ status: 'missing' });
          return;
        }
        setState({
          status: 'error',
          message: toUserMessage(error, { audience: 'professor', context: 'syllabus' }).message,
        });
      });
  }, [courseId]);

  // A new syllabus makes whatever was loaded stale; it is read again if open.
  useEffect(() => {
    latestRequest.current += 1;
    setState({ status: 'idle' });
  }, [courseId, refreshToken]);

  useEffect(() => {
    if (open && state.status === 'idle') {
      load();
    }
  }, [open, state.status, load]);

  return (
    <div className="syllabus-extracted">
      <Button
        variant="tertiary"
        size="sm"
        iconLeft="expand"
        aria-expanded={open}
        aria-controls={regionId}
        onClick={() => setOpen((current) => !current)}
      >
        {open ? 'Hide extracted text' : 'Show extracted text'}
      </Button>

      {open && (
        <div id={regionId} className="syllabus-extracted__region">
          {state.status === 'loading' && (
            <p className="ui-text-muted" role="status" aria-live="polite">
              Loading the extracted text…
            </p>
          )}
          {state.status === 'missing' && (
            <p className="ui-text-muted">No extracted text is stored for this course yet.</p>
          )}
          {state.status === 'error' && (
            <p className="ui-text-muted" role="alert">
              {state.message}{' '}
              <Button variant="tertiary" size="sm" onClick={load}>
                Try again
              </Button>
            </p>
          )}
          {state.status === 'ready' && (
            <>
              <p className="ui-text-xs ui-text-muted">
                {state.characterCount.toLocaleString()} characters, exactly as the course
                assistant reads them.
              </p>
              <pre
                className="syllabus-extracted__text"
                tabIndex={0}
                aria-label="Extracted syllabus text"
                data-testid="extracted-syllabus-text"
              >
                {state.text}
              </pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}
