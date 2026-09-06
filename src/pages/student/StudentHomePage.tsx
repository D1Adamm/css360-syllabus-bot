import { LinkButton } from '../../components/ui/Button';
import { Illustration } from '../../components/illustration/Illustration';
import { formatCourseCode } from '../../lib/courseLabels';
import { PageHeader } from '../../components/ui/PageHeader';
import { ProgressSteps } from '../../components/ui/ProgressSteps';
import { useCourseId } from '../../context/CourseContext';
import { useComparisonRunStore } from '../../context/ComparisonRunContext';
import { useCourseActivity } from '../../hooks/useCourseActivity';
import { useCourseMetadata } from '../../hooks/useCourseMetadata';
import { studentCoursePath } from '../../lib/roleRoutes';

/**
 * Student course home.
 *
 * The activity figures describe the course, not the reader — a student's
 * identity here is an anonymous participant id that nobody, including the
 * student, ever sees, and a personal score would be a step towards keeping
 * one. The figures are counts from the backend; no rating or contribution of
 * another student is downloaded to produce them. What is personalised is the
 * next action, which comes from this session: an unevaluated comparison is
 * the one thing the student is genuinely part-way through.
 */
export function StudentHomePage() {
  const courseId = useCourseId();
  const { metadata } = useCourseMetadata(courseId);
  const activity = useCourseActivity(courseId);
  const { getRun } = useComparisonRunStore();

  const pendingRun = getRun(courseId);

  const contributed =
    activity.status === 'ready' ? String(activity.activity.contributedQuestions) : '—';
  const evaluated =
    activity.status === 'ready' ? String(activity.activity.evaluations) : '—';

  const nextAction = pendingRun
    ? {
        step: 2,
        title: 'You have a comparison waiting',
        body: `You asked “${pendingRun.question}”. Rate those four answers to finish.`,
        to: studentCoursePath(courseId, 'evaluate'),
        label: 'Evaluate those responses',
      }
    : {
        step: 1,
        title: 'Ask a course question',
        body: 'See how four different AI approaches answer the same syllabus question.',
        to: studentCoursePath(courseId, 'compare'),
        label: 'Compare a question',
      };

  return (
    <div className="ui-stack ui-stack--section">
      <PageHeader
        eyebrow={
          metadata?.term
            ? `${metadata.term}${metadata.instructorName ? ` · ${metadata.instructorName}` : ''}`
            : undefined
        }
        title={formatCourseCode(metadata?.name) || 'Your course'}
        description="Help improve and evaluate an AI assistant for your course using questions grounded in the syllabus."
      />

      <section className="home__next" aria-labelledby="home-next-title">
        <div className="home__next-body">
          <p className="home__next-eyebrow">Next</p>
          <h2 className="home__next-title" id="home-next-title">
            {nextAction.title}
          </h2>
          <p className="home__next-text">{nextAction.body}</p>
          <LinkButton to={nextAction.to} variant="primary" iconRight="forward">
            {nextAction.label}
          </LinkButton>
        </div>
        <Illustration name="landing" size="md" className="home__next-art" />
      </section>

      <section className="ui-stack ui-stack--snug">
        <h2 className="home__section-title">How it works</h2>
        <ProgressSteps
          currentIndex={nextAction.step}
          // Highlights where you are without claiming a step is finished —
          // this is course-wide activity, not a personal record.
          statuses={
            nextAction.step === 2
              ? ['upcoming', 'upcoming', 'current']
              : ['upcoming', 'current', 'upcoming']
          }
          aria-label="Contribute, compare, evaluate"
          steps={[
            { id: 'contribute', label: 'Contribute', meta: 'Add a question' },
            { id: 'compare', label: 'Compare', meta: 'See four answers' },
            { id: 'evaluate', label: 'Evaluate', meta: 'Rate what you saw' },
          ]}
        />
      </section>

      <section className="ui-stack ui-stack--snug">
        <h2 className="home__section-title">This course so far</h2>
        <dl className="home__stats">
          <div className="home__stat">
            <dt>Questions contributed</dt>
            <dd>{contributed}</dd>
          </div>
          <div className="home__stat">
            <dt>Evaluations submitted</dt>
            <dd>{evaluated}</dd>
          </div>
        </dl>
        <p className="ui-text-xs ui-text-muted">
          Totals for the whole class. Contributions and ratings are anonymous.
        </p>
      </section>

      <section className="home__links">
        <LinkButton
          to={studentCoursePath(courseId, 'contribute')}
          variant="secondary"
          iconLeft="contribute"
        >
          Contribute a question
        </LinkButton>
        <LinkButton
          to={studentCoursePath(courseId, 'syllabus')}
          variant="tertiary"
          iconLeft="syllabus"
        >
          Read the syllabus
        </LinkButton>
      </section>
    </div>
  );
}
