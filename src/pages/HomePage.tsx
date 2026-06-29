import { ButtonLink } from '../components/ButtonLink';
import { SectionCard } from '../components/SectionCard';

const workflowSteps = [
  'Syllabus',
  'Seed Examples',
  'Fine-Tuning',
  'Model Comparison',
  'Evaluation',
];

export function HomePage() {
  return (
    <>
      <section className="hero" aria-labelledby="hero-title">
        <h1 id="hero-title" className="hero__title">
          Syllabus Model Lab
        </h1>
        <p className="hero__subtitle">
          A classroom prototype for comparing how base models, retrieval-augmented
          generation (RAG), fine-tuning, and fine-tuning combined with RAG answer
          questions about a course syllabus.
        </p>
        <div className="hero__actions">
          <ButtonLink to="/seed-builder">Seed Data Builder</ButtonLink>
          <ButtonLink to="/compare" variant="secondary">
            Model Comparison
          </ButtonLink>
        </div>
      </section>

      <section className="workflow" aria-labelledby="workflow-title">
        <h2 id="workflow-title" className="workflow__title">
          Classroom Workflow
        </h2>
        <div className="workflow__steps" role="list" aria-label="Workflow steps">
          {workflowSteps.map((step, index) => (
            <span key={step} role="listitem" style={{ display: 'contents' }}>
              <span className="workflow__step">{step}</span>
              {index < workflowSteps.length - 1 && (
                <span className="workflow__arrow" aria-hidden="true">
                  →
                </span>
              )}
            </span>
          ))}
        </div>
      </section>

      <section className="approach" aria-labelledby="approach-title">
        <h2 id="approach-title" className="approach__title">
          Comparison Approaches
        </h2>
        <div className="approach__grid">
          <SectionCard title="Base Model">
            A general model answering without course-specific syllabus context.
          </SectionCard>
          <SectionCard title="RAG">
            A model that receives relevant syllabus passages when answering.
          </SectionCard>
          <SectionCard title="Fine-Tuned Model">
            A model adjusted using student-created question-and-answer examples.
          </SectionCard>
          <SectionCard title="Fine-Tuned + RAG">
            A fine-tuned model that also receives retrieved syllabus context.
          </SectionCard>
        </div>
      </section>

      <section className="scope-notice" aria-labelledby="scope-title">
        <h2 id="scope-title" className="scope-notice__title">
          Prototype Scope
        </h2>
        <p className="scope-notice__text">
          This application is currently a frontend prototype. Navigation, layout,
          and placeholder pages are in place, but live model inference, training
          pipelines, and data persistence are not running yet. Future phases will
          add syllabus exploration, seed data creation, model comparison, and
          evaluation features.
        </p>
      </section>
    </>
  );
}
