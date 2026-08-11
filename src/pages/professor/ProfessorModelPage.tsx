import { LinkButton } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { Illustration } from '../../components/illustration/Illustration';
import { formatCourseCode } from '../../lib/courseLabels';
import { PageHeader } from '../../components/ui/PageHeader';
import { StatusPill } from '../../components/ui/StatusPill';
import { useCourseId } from '../../context/CourseContext';
import { useCourseExampleCounts } from '../../hooks/useCourseExampleCounts';
import { useCourseMetadata } from '../../hooks/useCourseMetadata';
import {
  getCourseModelStatus,
  getModelReadiness,
  RECOMMENDED_APPROVED_EXAMPLES,
} from '../../lib/modelStatus';
import { professorCoursePath } from '../../lib/roleRoutes';

/**
 * Course model status — an integration boundary.
 *
 * Requesting a model has no backend: there is no request record, no training
 * job status, and no per-course model registry. So this page reports the one
 * thing it can verify — how many approved examples exist — and says plainly
 * that requesting is not available yet. No button pretends to submit anything.
 *
 * All the guesswork is confined to `lib/modelStatus.ts`. See
 * `docs/frontend-backend-gaps.md` for the endpoints this needs.
 */
export function ProfessorModelPage() {
  const courseId = useCourseId();
  const { metadata } = useCourseMetadata(courseId);
  const countsState = useCourseExampleCounts(courseId);

  const counts = countsState.status === 'ready' ? countsState.counts : null;
  const readiness = getModelReadiness(counts?.approved ?? 0);
  const status = getCourseModelStatus();

  return (
    <div className="ui-stack ui-stack--section">
      <PageHeader
        eyebrow={formatCourseCode(metadata?.name)}
        title="Course model"
        description="A course model learns from the examples you approve, so the assistant can answer in your course's own words."
      />

      <section className="model-state">
        <Illustration name="model-ready" size="lg" />

        <div className="model-state__body">
          <StatusPill tone="neutral">Not available yet</StatusPill>

          <h2 className="model-state__title">
            {countsState.status === 'loading'
              ? 'Checking your examples…'
              : counts
                ? `You have ${counts.approved} approved example${counts.approved === 1 ? '' : 's'}.`
                : 'Your approved examples could not be counted right now.'}
          </h2>

          <p className="model-state__text">
            {counts && !readiness.hasEnough
              ? `Around ${RECOMMENDED_APPROVED_EXAMPLES} approved examples make a course model worthwhile — about ${readiness.remaining} more to go.`
              : counts && readiness.hasEnough
                ? 'That is enough to train a course model. Requesting one is not available yet.'
                : 'Approve examples in the review queue to build up a set for training.'}
          </p>

          <div className="model-state__actions">
            <LinkButton
              to={professorCoursePath(courseId, 'examples')}
              variant="primary"
              iconRight="forward"
            >
              Review examples
            </LinkButton>
          </div>
        </div>
      </section>

      {status.isPlaceholder && (
        <Callout tone="info" title="Requesting a course model isn't built yet">
          Your approved examples are saved and will be used when this part of the
          project is ready. Nothing is lost in the meantime — keep reviewing, and
          the set will be waiting.
        </Callout>
      )}
    </div>
  );
}
