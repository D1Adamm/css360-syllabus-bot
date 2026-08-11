import { LinkButton } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { ErrorState } from '../../components/ui/ErrorState';
import { Illustration } from '../../components/illustration/Illustration';
import { PageHeader } from '../../components/ui/PageHeader';
import { StatusPill } from '../../components/ui/StatusPill';
import { useCourseId } from '../../context/CourseContext';
import { useCourseExampleCounts } from '../../hooks/useCourseExampleCounts';
import { useCourseMetadata } from '../../hooks/useCourseMetadata';
import { useCourseModel } from '../../hooks/useCourseModel';
import { formatCourseCode } from '../../lib/courseLabels';
import { getCurrentVersion } from '../../lib/courseModelDb';
import {
  describeCourseModel,
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
 * Still absent: requesting training from the UI. There is no request record and
 * no job submission endpoint, so no button pretends to start one.
 */
export function ProfessorModelPage() {
  const courseId = useCourseId();
  const { metadata } = useCourseMetadata(courseId);
  const countsState = useCourseExampleCounts(courseId);
  const { state: modelState, retry } = useCourseModel(courseId);

  const counts = countsState.status === 'ready' ? countsState.counts : null;
  const readiness = getModelReadiness(counts?.approved ?? 0);

  const version =
    modelState.status === 'ready' ? getCurrentVersion(modelState.registry) : null;

  const presentation = describeCourseModel({
    version,
    registryUnavailable: modelState.status === 'unavailable',
  });

  const hasModel = presentation.presence === 'ready';

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
            <StatusPill tone={presentation.tone}>
              {presentation.presence === 'ready'
                ? presentation.availability === 'online'
                  ? 'Ready · in use'
                  : presentation.availability === 'offline'
                    ? 'Ready · offline'
                    : 'Ready'
                : presentation.presence === 'training'
                  ? 'Preparing'
                  : presentation.presence === 'failed'
                    ? 'Needs attention'
                    : presentation.presence === 'none'
                      ? 'Not created yet'
                      : 'Unknown'}
            </StatusPill>

            <h2 className="model-state__title">{presentation.title}</h2>
            <p className="model-state__text">{presentation.detail}</p>

            {/* What the model was built from. No artifact reference, no base
                model id, no service address — none of it changes what a
                professor would do. */}
            {hasModel && version && (
              <dl className="model-facts">
                <div>
                  <dt>Trained from</dt>
                  <dd>
                    {version.trainingExampleCount} approved example
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

            {!hasModel && (
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
              <LinkButton
                to={professorCoursePath(courseId, 'examples')}
                variant={hasModel ? 'secondary' : 'primary'}
                iconRight="forward"
              >
                Review examples
              </LinkButton>
            </div>
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

      {presentation.presence === 'none' && (
        <Callout tone="info" title="Requesting a model isn't built yet">
          Training is started by the project administrator for now. Your approved
          examples are saved and will be used when it runs — keep reviewing, and
          the set will be waiting.
        </Callout>
      )}
    </div>
  );
}
