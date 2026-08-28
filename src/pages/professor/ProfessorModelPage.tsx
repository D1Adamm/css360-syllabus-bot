import { Button, LinkButton } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { ErrorState } from '../../components/ui/ErrorState';
import { Illustration } from '../../components/illustration/Illustration';
import { PageHeader } from '../../components/ui/PageHeader';
import { StatusPill } from '../../components/ui/StatusPill';
import { useCourseId } from '../../context/CourseContext';
import { useCourseExampleCounts } from '../../hooks/useCourseExampleCounts';
import { useCourseMetadata } from '../../hooks/useCourseMetadata';
import { useCourseModel } from '../../hooks/useCourseModel';
import { useCourseModelRequest } from '../../hooks/useCourseModelRequest';
import { formatCourseCode } from '../../lib/courseLabels';
import { getCurrentVersion } from '../../lib/courseModelDb';
import {
  describeCourseModel,
  describeModelRequest,
  getModelReadiness,
  RECOMMENDED_APPROVED_EXAMPLES,
} from '../../lib/modelStatus';
import { professorCoursePath } from '../../lib/roleRoutes';

/**
 * Whether this course has a model, and whether it can answer right now.
 *
 * Those are separate facts and the page says both. A model trained from a
 * professor's approved examples does not stop existing because the inference
 * service is stopped — the previous version of this page conflated the two and
 * so told CSS 360 it had no model at all.
 *
 * Nothing here reads `/fine-tuned/health`: that endpoint describes one shared
 * service and cannot say which course's adapter is loaded. Existence comes
 * only from this course's registry record.
 *
 * A professor whose course has no model, and enough approved examples, can
 * request one. That writes a durable row in `model_requests`
 * — separate from the registry, because a request is work asked for and the
 * registry is artifacts that exist. Nothing here starts a training run; the
 * request is a queue entry an administrator picks up.
 */
export function ProfessorModelPage() {
  const courseId = useCourseId();
  const { metadata } = useCourseMetadata(courseId);
  const countsState = useCourseExampleCounts(courseId);
  const { state: modelState, retry } = useCourseModel(courseId);
  const {
    state: requestState,
    submitting,
    submitError,
    submit,
  } = useCourseModelRequest(courseId);

  const counts = countsState.status === 'ready' ? countsState.counts : null;
  const readiness = getModelReadiness(counts?.approved ?? 0);

  const version =
    modelState.status === 'ready' ? getCurrentVersion(modelState.registry) : null;

  const presentation = describeCourseModel({
    version,
    registryUnavailable: modelState.status === 'unavailable',
  });

  const hasModel = presentation.presence === 'ready';

  const request = requestState.status === 'ready' ? requestState.request : null;
  // A terminal request tells the professor nothing they cannot see from the
  // model itself, so only outstanding work is surfaced.
  const outstandingRequest =
    request && request.status !== 'ready' && request.status !== 'failed'
      ? request
      : null;
  const failedRequest = request?.status === 'failed' ? request : null;

  /*
   * Whether outstanding work should replace the headline, or sit under it.
   *
   * It replaces the headline only when there is no model yet — then "being
   * prepared" is the whole story. Once a course has a ready model, a second
   * one being built must not take over the page: the model they have is still
   * registered, still theirs, and still answering questions. Saying "your
   * course model is training" there would be a page telling a professor they
   * have nothing while the thing they have keeps working.
   *
   * This became reachable when administrators gained a Train new version
   * action. Before it, a ready course could never have outstanding work.
   */
  const buildingNewVersion = hasModel && outstandingRequest !== null;
  const activeRequest = buildingNewVersion ? null : outstandingRequest;

  /*
   * The Request button appears only when all four are true:
   *   - the registry has been read and this course has no model
   *   - enough approved examples to be worth training on
   *   - no request is already outstanding
   *   - the request record itself was readable
   *
   * CSS 360 fails the first, so its page keeps the Ready/Offline treatment and
   * never offers a first-model request. Retraining is a separate feature.
   */
  const canRequest =
    modelState.status !== 'loading' &&
    modelState.status !== 'unavailable' &&
    presentation.presence === 'none' &&
    readiness.hasEnough &&
    requestState.status === 'none';

  return (
    <div className="ui-stack ui-stack--section">
      <PageHeader
        eyebrow={formatCourseCode(metadata?.name)}
        title="Course model"
        description="A course model learns from the examples you approve, so the assistant can answer in your course's own words."
      />

      {modelState.status === 'unavailable' && (
        <ErrorState
          title="Model status unavailable"
          message="We couldn't check your course model just now. Nothing about it has changed."
          onRetry={retry}
        />
      )}

      {modelState.status === 'loading' ? (
        <p className="ui-text-muted" role="status" aria-live="polite">
          Checking your course model…
        </p>
      ) : (
        <section className="model-state">
          <Illustration name="model-ready" size="lg" />

          <div className="model-state__body">
            <StatusPill tone={activeRequest ? describeModelRequest(activeRequest).tone : presentation.tone}>
              {activeRequest
                ? describeModelRequest(activeRequest).label
                : presentation.presence === 'ready'
                ? presentation.availability === 'online'
                  ? 'Ready · published'
                  : presentation.availability === 'offline'
                    ? 'Ready · not published'
                    : 'Ready'
                : presentation.presence === 'training'
                  ? 'Preparing'
                  : presentation.presence === 'failed'
                    ? 'Needs attention'
                    : presentation.presence === 'none'
                      ? 'Not created yet'
                      : 'Unknown'}
            </StatusPill>

            <h2 className="model-state__title">
              {activeRequest ? describeModelRequest(activeRequest).title : presentation.title}
            </h2>
            <p className="model-state__text">
              {activeRequest
                ? describeModelRequest(activeRequest).detail
                : presentation.detail}
            </p>

            {activeRequest && (
              <p className="ui-text-xs ui-text-muted">
                Requested {new Date(activeRequest.requestedAt).toLocaleDateString()} ·{' '}
                {activeRequest.approvedExampleCount} approved example
                {activeRequest.approvedExampleCount === 1 ? '' : 's'} at the time
              </p>
            )}

            {/* A new version being built is an addition to what they have,
                not a replacement for it, so it reads as one quiet line under
                the model they are still using. */}
            {buildingNewVersion && (
              <p className="ui-text-xs ui-text-muted" role="status">
                An updated version is being prepared from your approved
                examples. The model above keeps working until it is ready.
              </p>
            )}

            {/* What the model was built from. No artifact reference, no base
                model id, no service address — none of it changes what a
                professor would do. */}
            {hasModel && version && (
              <dl className="model-facts">
                <div>
                  {/* "approved examples" was wrong here: the count is the train
                      split, which is a subset of what was approved. A professor
                      who approved 42 and read "42 approved examples" against a
                      model trained on 37 was being told something untrue about
                      their own work. */}
                  <dt>Trained on</dt>
                  <dd>
                    {version.trainingExampleCount} example
                    {version.trainingExampleCount === 1 ? '' : 's'}
                  </dd>
                </div>
                <div>
                  <dt>Version</dt>
                  <dd>{version.version}</dd>
                </div>
                <div>
                  <dt>Prepared</dt>
                  <dd>{new Date(version.createdAt).toLocaleDateString()}</dd>
                </div>
              </dl>
            )}

            {!hasModel && !activeRequest && (
              <p className="model-state__text">
                {countsState.status === 'loading'
                  ? 'Counting your approved examples…'
                  : counts
                    ? readiness.hasEnough
                      ? `You have ${counts.approved} approved examples — enough to train one.`
                      : `You have ${counts.approved} approved. Around ${RECOMMENDED_APPROVED_EXAMPLES} makes a course model worthwhile — about ${readiness.remaining} more to go.`
                    : 'Your approved examples could not be counted right now.'}
              </p>
            )}

            <div className="model-state__actions">
              {canRequest && (
                <Button
                  variant="primary"
                  iconLeft="model"
                  loading={submitting}
                  loadingLabel="Sending…"
                  onClick={() => void submit(counts?.approved ?? 0)}
                >
                  Request course model
                </Button>
              )}
              <LinkButton
                to={professorCoursePath(courseId, 'examples')}
                variant={hasModel || canRequest ? 'secondary' : 'primary'}
                iconRight="forward"
              >
                Review examples
              </LinkButton>
            </div>

            {submitError && (
              <p className="model-state__error" role="alert">
                {submitError}
              </p>
            )}
          </div>
        </section>
      )}

      {hasModel && presentation.availability === 'offline' && (
        <Callout tone="info" title="Why it's offline">
          Course models run on shared research hardware that isn't kept running
          continuously. Your model and its training examples are saved. Contact
          the project administrator when you'd like it available for your
          students.
        </Callout>
      )}

      {presentation.presence === 'none' && !activeRequest && !readiness.hasEnough && (
        <Callout tone="info" title="Keep reviewing">
          A course model is worth training once around{' '}
          {RECOMMENDED_APPROVED_EXAMPLES} examples are approved. Everything you
          approve is saved and will be used when you request one.
        </Callout>
      )}

      {failedRequest && (
        <Callout tone="warning" title="Your last request didn't complete">
          The project administrator has the details and will follow up. Your
          approved examples are safe, and you can request again once they have
          looked at it.
        </Callout>
      )}
    </div>
  );
}
