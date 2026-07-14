import { PageHeader } from '../components/PageHeader';

export function ArchitecturePage() {
  return (
    <>
      <PageHeader
        title="Architecture"
        description="Understand how the Syllabus Model Lab multi-course prototype is structured and what is live versus simulated."
      />

      <section className="architecture-section" aria-labelledby="arch-frontend-title">
        <h2 id="arch-frontend-title" className="architecture-section__title">
          Current frontend
        </h2>
        <p className="architecture-section__text">
          The application is a React + TypeScript single-page app bundled by Vite and deployable on
          Firebase Hosting. React Router serves a course picker at <code>/</code>, course pages under{' '}
          <code>/course/:courseId/...</code>, plus <code>/create-course</code> and{' '}
          <code>/architecture</code>. Course metadata, seed examples, and evaluations live in Firebase
          Realtime Database under <code>courses/{'{courseId}'}</code>.
        </p>
        <ul className="architecture-section__list">
          <li>React 19 with functional components and hooks</li>
          <li>TypeScript for shared data types and utilities</li>
          <li>Vite for development and production builds</li>
          <li>Firebase Hosting for the static frontend</li>
          <li>Firebase Realtime Database for course-scoped data</li>
          <li>Plain CSS with no component library</li>
        </ul>
      </section>

      <section className="architecture-section" aria-labelledby="arch-courses-title">
        <h2 id="arch-courses-title" className="architecture-section__title">
          Multi-course data model
        </h2>
        <p className="architecture-section__text">
          Each course is keyed by a stable <code>courseId</code>. Firebase stores metadata, seed
          examples, and evaluations per course. Syllabus files and embedding indexes are stored
          locally by the FastAPI backend for now.
        </p>
        <ul className="architecture-section__list">
          <li>
            Firebase: <code>courses/{'{courseId}'}/metadata|seedExamples|evaluations</code>
          </li>
          <li>
            Local syllabus text: <code>backend/course_data/{'{courseId}'}/syllabus.txt</code>
          </li>
          <li>
            Local RAG index: <code>backend/data/indexes/{'{courseId}'}.json</code>
          </li>
        </ul>
      </section>

      <section className="architecture-section" aria-labelledby="arch-backend-title">
        <h2 id="arch-backend-title" className="architecture-section__title">
          Backend and models
        </h2>
        <p className="architecture-section__text">
          A FastAPI backend handles syllabus upload/extraction, chunking, embeddings with{' '}
          <code>nomic-embed-text</code>, course-specific RAG retrieval, and Base Model generation with
          Ollama <code>llama3.2:3b</code>. Fine-Tuned and Fine-Tuned + RAG comparison cards remain
          simulated from static prototype data.
        </p>
      </section>

      <section className="architecture-section" aria-labelledby="arch-flow-title">
        <h2 id="arch-flow-title" className="architecture-section__title">
          Current data flow
        </h2>
        <div className="architecture-flow" role="img" aria-label="Data flow diagram">
          <div className="architecture-flow__step">Create course + upload syllabus</div>
          <div className="architecture-flow__arrow" aria-hidden="true">↓</div>
          <div className="architecture-flow__step">Extract text, chunk, embed course index</div>
          <div className="architecture-flow__arrow" aria-hidden="true">↓</div>
          <div className="architecture-flow__step">Course-scoped seed examples in Firebase</div>
          <div className="architecture-flow__arrow" aria-hidden="true">↓</div>
          <div className="architecture-flow__step">
            Live Base / course RAG + simulated Fine-Tuned cards
          </div>
          <div className="architecture-flow__arrow" aria-hidden="true">↓</div>
          <div className="architecture-flow__step">Course-scoped evaluations in Firebase</div>
          <div className="architecture-flow__arrow" aria-hidden="true">↓</div>
          <div className="architecture-flow__step">Results dashboard</div>
        </div>
      </section>

      <section className="architecture-section" aria-labelledby="arch-future-title">
        <h2 id="arch-future-title" className="architecture-section__title">
          Future work
        </h2>
        <p className="architecture-section__text">
          Phase 1 keeps syllabus artifacts on local disk. A later phase can move artifact storage to
          GCP or a VM, add instructor authentication, and replace simulated fine-tuned responses with
          trained models. Root-level legacy Firebase collections (<code>seedExamples</code>,{' '}
          <code>evaluations</code>) may still exist in the database and can be deleted manually after
          confirming they are unused.
        </p>
      </section>
    </>
  );
}
