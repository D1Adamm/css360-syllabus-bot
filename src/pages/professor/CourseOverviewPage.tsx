import { Link } from 'react-router-dom';
import { LinkButton } from '../../components/ui/Button';
import { ErrorState } from '../../components/ui/ErrorState';
import { formatCourseCode } from '../../lib/courseLabels';
import { PageHeader } from '../../components/ui/PageHeader';
import { StatusPill } from '../../components/ui/StatusPill';
import { useCourseId } from '../../context/CourseContext';
import { useCourseExampleCounts } from '../../hooks/useCourseExampleCounts';
import {
  useCourseMetadata,
  type CourseMetadataState,
} from '../../hooks/useCourseMetadata';
import { useCourseModel } from '../../hooks/useCourseModel';
import { useCourseModelRequest } from '../../hooks/useCourseModelRequest';
import { useEvaluations } from '../../hooks/useEvaluations';
import { getCurrentVersion } from '../../lib/courseModelDb';
import { canRequestCourseModel, summariseCourseModel } from '../../lib/modelStatus';
import { professorCoursePath } from '../../lib/roleRoutes';
import { getStarterGeneration } from '../../lib/starterSeedGeneration';
import type { SyllabusStatus } from '../../types';

interface Presentation {
  label: string;
  tone: 'neutral' | 'success' | 'warning' | 'danger' | 'progress';
}

/**
 * Durable syllabus state, not connectivity.
 *
 * `metadataState` decides first: while the record is loading, or when it could
 * not be read, we do not know anything about the syllabus and must not claim
 * it is missing. Only once we have the record does its status decide.
 */
function syllabusPresentation(
  metadataState: CourseMetadataState,
): Presentation {
  if (metadataState.status === 'loading') {
    return { label: 'Checking…', tone: 'neutral' };
  }
  if (metadataState.status === 'unavailable') {
    return { label: 'Temporarily unavailable', tone: 'warning' };
  }
  if (metadataState.status === 'missing') {
    return { label: 'Not added yet', tone: 'neutral' };
  }
  return syllabusStatusPresentation(metadataState.metadata.syllabusStatus);
}

function syllabusStatusPresentation(status: SyllabusStatus | undefined): Presentation {
  switch (status) {
    case 'indexed':
    case 'ready':
      return { label: 'Ready', tone: 'success' };
    case 'uploaded':
    case 'extracted':
    case 'processing':
      return { label: 'Preparing', tone: 'progress' };
    case 'upload_failed':
    case 'index_failed':
    case 'error':
      return { label: 'Needs attention', tone: 'danger' };
    default:
      return { label: 'Not added yet', tone: 'neutral' };
  }
}

/**
 * At-a-glance state for one course.
 *
 * Four rows, not a dashboard: a professor arriving here wants to know whether
 * anything needs them. When something does, it is called out above the fold;
 * when nothing does, that section disappears rather than showing a row of
 * zeroes.
 */
export function CourseOverviewPage() {
  const courseId = useCourseId();
  const { state: metadataState, metadata, retry: retryMetadata } = useCourseMetadata(courseId);
  const countsState = useCourseExampleCounts(courseId);
  const { state: modelState } = useCourseModel(courseId);
  const { state: requestState } = useCourseModelRequest(courseId);
  const { evaluations } = useEvaluations();

  const counts = countsState.status === 'ready' ? countsState.counts : null;

  /*
   * The real model state for this course, from the same two records and the
   * same helper the Model page uses.
   *
   * This row used to be a hardcoded "Not available yet", which was wrong for
   * every course that had requested, trained, or published anything — and said
   * the opposite of the Model page one click away. Both hooks are course-scoped
   * and re-subscribe when `courseId` changes, so one course's state cannot be
   * shown against another's.
   */
  const version =
    modelState.status === 'ready' ? getCurrentVersion(modelState.registry) : null;
  const request = requestState.status === 'ready' ? requestState.request : null;

  const model = summariseCourseModel({
    version,
    request,
    loading:
      modelState.status === 'loading' || requestState.status === 'loading',
    registryUnavailable: modelState.status === 'unavailable',
    requestUnavailable: requestState.status === 'unavailable',
  });
  const syllabus = syllabusPresentation(metadataState);

  /*
   * Whether a first model can be asked for, from the same helper that decides
   * whether the Model page offers the button.
   *
   * The approved count alone used to decide it, so a course whose model was
   * trained, registered and published still carried "you have 42 approved
   * examples — enough for a course model" under NEEDS YOUR ATTENTION, directly
   * above a Course model row reading "Ready · published". Nothing needed the
   * professor's attention. Attention is now only claimed when the action behind
   * it actually exists.
   */
  const canRequestModel = canRequestCourseModel({
    version,
    request,
    approved: counts?.approved ?? 0,
    registryLoading: modelState.status === 'loading',
    registryUnavailable: modelState.status === 'unavailable',
    requestLoading: requestState.status === 'loading',
    requestUnavailable: requestState.status === 'unavailable',
  });

  /*
   * Starter examples are still being written for this course.
   *
   * While that is true, "0 approved" is not a fact about the course — it is a
   * fact about a job that has not finished. Reporting it as though the course
   * had nothing is what made an upload look like it had done nothing at all.
   */
  const generating = getStarterGeneration(metadata).state === 'generating';

  const attention: { key: string; text: string; to: string; action: string }[] = [];

  if (counts && counts.pending > 0) {
    attention.push({
      key: 'pending',
      text: `${counts.pending} example${counts.pending === 1 ? '' : 's'} waiting for your review`,
      to: professorCoursePath(courseId, 'examples'),
      action: 'Review examples',
    });
  }

  if (
    metadataState.status === 'ready' &&
    syllabus.tone !== 'success' &&
    syllabus.tone !== 'progress'
  ) {
    attention.push({
      key: 'syllabus',
      text: 'This course has no syllabus ready yet',
      to: professorCoursePath(courseId, 'syllabus'),
      action: 'View syllabus',
    });
  }

  if (counts && canRequestModel) {
    attention.push({
      key: 'model',
      text: `You have ${counts.approved} approved examples — enough to request a course model`,
      to: professorCoursePath(courseId, 'model'),
      action: 'See course model',
    });
  }

  return (
    <div className="ui-stack ui-stack--section">
      <PageHeader
        eyebrow={metadata?.term}
        title={formatCourseCode(metadata?.name) || 'Course'}
        description={metadata?.title}
        actions={
          <LinkButton
            to={professorCoursePath(courseId, 'invite')}
            variant="secondary"
            iconLeft="students"
          >
            Invite students
          </LinkButton>
        }
      />

      {attention.length > 0 && (
        <section className="attention" aria-labelledby="attention-title">
          <h2 className="attention__title" id="attention-title">
            Needs your attention
          </h2>
          <ul className="attention__list">
            {attention.map((item) => (
              <li key={item.key} className="attention__item">
                <span className="attention__text">{item.text}</span>
                <Link to={item.to} className="attention__action">
                  {item.action} →
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}

      {metadataState.status === 'unavailable' && (
        <ErrorState
          title="Course details unavailable"
          message="We couldn't load this course's details just now. Your course and its content are unaffected."
          onRetry={retryMetadata}
        />
      )}

      <section className="ui-stack ui-stack--snug">
        <h2 className="overview__title">Course status</h2>
        <dl className="overview">
          <div className="overview__row">
            <dt className="overview__label">Syllabus</dt>
            <dd className="overview__value">
              <StatusPill tone={syllabus.tone}>{syllabus.label}</StatusPill>
            </dd>
            <Link
              to={professorCoursePath(courseId, 'syllabus')}
              className="overview__link"
            >
              View
            </Link>
          </div>

          <div className="overview__row">
            <dt className="overview__label">Training examples</dt>
            <dd className="overview__value">
              {generating ? (
                <>
                  <StatusPill tone="progress">Generating…</StatusPill>
                  {/* Anything already reviewable is still worth saying: it is
                      work a professor can start on now. */}
                  {counts && counts.pending > 0 && (
                    <span className="overview__muted">
                      {' '}
                      · {counts.pending} ready to review
                    </span>
                  )}
                </>
              ) : counts ? (
                <>
                  <strong>{counts.approved}</strong> approved
                  {counts.pending > 0 && (
                    <span className="overview__muted">
                      {' '}
                      · {counts.pending} awaiting review
                    </span>
                  )}
                </>
              ) : countsState.status === 'loading' ? (
                <span className="overview__muted">Loading…</span>
              ) : (
                <span className="overview__muted">Not available right now</span>
              )}
            </dd>
            <Link
              to={professorCoursePath(courseId, 'examples')}
              className="overview__link"
            >
              Review
            </Link>
          </div>

          <div className="overview__row">
            <dt className="overview__label">Course model</dt>
            <dd className="overview__value">
              <StatusPill tone={model.tone}>{model.label}</StatusPill>
            </dd>
            <Link to={professorCoursePath(courseId, 'model')} className="overview__link">
              Details
            </Link>
          </div>

          <div className="overview__row">
            <dt className="overview__label">Student evaluations</dt>
            <dd className="overview__value">
              <strong>{evaluations.length}</strong> received
            </dd>
            <Link
              to={professorCoursePath(courseId, 'results')}
              className="overview__link"
            >
              View results
            </Link>
          </div>
        </dl>
      </section>
    </div>
  );
}
