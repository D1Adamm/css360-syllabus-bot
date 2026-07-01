import { PageHeader } from '../components/PageHeader';

export function ArchitecturePage() {
  return (
    <>
      <PageHeader
        title="Architecture"
        description="Understand how the Syllabus Model Lab prototype is structured technically and what is simulated versus implemented."
      />

      <section className="architecture-section" aria-labelledby="arch-frontend-title">
        <h2 id="arch-frontend-title" className="architecture-section__title">
          Current frontend
        </h2>
        <p className="architecture-section__text">
          The application is a single-page React app built with TypeScript and bundled by Vite.
          React Router handles client-side navigation between classroom workflow pages. All
          prototype syllabus, seed, and comparison data ships as static JSON files imported at
          build time. User-created seed examples and evaluation records persist in the browser
          through localStorage. Styling uses plain CSS with no component library or charting
          dependency.
        </p>
        <ul className="architecture-section__list">
          <li>React 19 with functional components and hooks</li>
          <li>TypeScript for shared data types and utility functions</li>
          <li>Vite for development server and production builds</li>
          <li>React Router for routes such as <code>/compare</code> and <code>/evaluate</code></li>
          <li>Static JSON data files under <code>src/data/</code></li>
          <li>localStorage for user seeds and evaluation records</li>
        </ul>
      </section>

      <section className="architecture-section" aria-labelledby="arch-syllabus-title">
        <h2 id="arch-syllabus-title" className="architecture-section__title">
          Syllabus data
        </h2>
        <p className="architecture-section__text">
          <code>docs/syllabus.txt</code> is the authoritative source document for the CSS 360
          course syllabus. It is preserved as plain text and is not modified by the application.
          <code>src/data/syllabusTopics.json</code> provides a structured summary of syllabus
          topics used by the Syllabus Explorer, including categories, summaries, source sections,
          and detail bullets for classroom browsing.
        </p>
      </section>

      <section className="architecture-section" aria-labelledby="arch-seed-title">
        <h2 id="arch-seed-title" className="architecture-section__title">
          Seed data
        </h2>
        <p className="architecture-section__text">
          Prototype seed examples live in <code>src/data/seedData.json</code> as instruction–
          response pairs derived from the syllabus. Students can create additional examples on
          the Seed Data Builder page; those are stored under the localStorage key{' '}
          <code>syllabus-demo-user-seeds</code>. The Dataset page combines prototype and
          user-created examples, supports filtering, and offers JSON and JSONL export for
          classroom review.
        </p>
      </section>

      <section className="architecture-section" aria-labelledby="arch-comparison-title">
        <h2 id="arch-comparison-title" className="architecture-section__title">
          Model comparison outputs
        </h2>
        <p className="architecture-section__text">
          Comparison records in <code>src/data/comparisonData.json</code> contain pre-written
          responses for Fine-Tuned and Fine-Tuned + RAG. Base Model and RAG responses on the
          Model Comparison page are live from the local FastAPI backend. Grounding labels (Low,
          Medium, High) on simulated cards are prototype annotations to support classroom
          discussion, not automated measurements from a retrieval or evaluation pipeline.
        </p>
      </section>

      <section className="architecture-section" aria-labelledby="arch-evaluation-title">
        <h2 id="arch-evaluation-title" className="architecture-section__title">
          Evaluation data
        </h2>
        <p className="architecture-section__text">
          When students submit ratings on the Evaluation page, records are appended to
          localStorage under <code>syllabus-demo-evaluations</code>. The Results dashboard
          aggregates those records in the browser. No server receives evaluation data. Multiple
          evaluations for the same question are allowed to simulate several classroom responses
          in one browser session.
        </p>
      </section>

      <section className="architecture-section" aria-labelledby="arch-flow-title">
        <h2 id="arch-flow-title" className="architecture-section__title">
          Current data flow
        </h2>
        <div className="architecture-flow" role="img" aria-label="Data flow diagram">
          <div className="architecture-flow__step">Official syllabus text</div>
          <div className="architecture-flow__arrow" aria-hidden="true">↓</div>
          <div className="architecture-flow__step">Structured syllabus topics</div>
          <div className="architecture-flow__arrow" aria-hidden="true">↓</div>
          <div className="architecture-flow__step">
            Prototype and user-created seed examples
          </div>
          <div className="architecture-flow__arrow" aria-hidden="true">↓</div>
          <div className="architecture-flow__step">
            Live Base Model / RAG plus simulated Fine-Tuned comparisons
          </div>
          <div className="architecture-flow__arrow" aria-hidden="true">↓</div>
          <div className="architecture-flow__step">Local evaluations</div>
          <div className="architecture-flow__arrow" aria-hidden="true">↓</div>
          <div className="architecture-flow__step">Results dashboard</div>
        </div>
      </section>

      <section className="architecture-section" aria-labelledby="arch-future-title">
        <h2 id="arch-future-title" className="architecture-section__title">
          Future backend boundary
        </h2>
        <p className="architecture-section__text">
          A production classroom research system would separate the frontend from model services
          and persistent storage:
        </p>
        <div className="architecture-flow architecture-flow--future" role="img" aria-label="Future architecture diagram">
          <div className="architecture-flow__step">Frontend</div>
          <div className="architecture-flow__arrow" aria-hidden="true">↓</div>
          <div className="architecture-flow__step">API layer</div>
          <div className="architecture-flow__arrow" aria-hidden="true">↓</div>
          <div className="architecture-flow__step">
            Base model / RAG / Fine-Tuned model / Fine-Tuned + RAG
          </div>
          <div className="architecture-flow__arrow" aria-hidden="true">↓</div>
          <div className="architecture-flow__step">Database and evaluation storage</div>
        </div>
      </section>

      <section className="architecture-section" aria-labelledby="arch-real-sim-title">
        <h2 id="arch-real-sim-title" className="architecture-section__title">
          What is real versus simulated
        </h2>
        <div className="architecture-comparison">
          <div className="architecture-comparison__column">
            <h3 className="architecture-comparison__heading">Real (implemented)</h3>
            <ul className="architecture-section__list">
              <li>Navigation and page layout</li>
              <li>Syllabus topic exploration and search</li>
              <li>Seed example creation and validation</li>
              <li>localStorage persistence for user seeds and evaluations</li>
              <li>Dataset filtering, sorting, and statistics</li>
              <li>JSON and JSONL export</li>
              <li>Model comparison interface with question selection</li>
              <li>Live Base Model and RAG inference via local FastAPI backend</li>
              <li>Local syllabus retrieval index and cosine-similarity search</li>
              <li>Evaluation workflow with form validation</li>
              <li>Results aggregation and export</li>
            </ul>
          </div>
          <div className="architecture-comparison__column">
            <h3 className="architecture-comparison__heading">Simulated (not implemented)</h3>
            <ul className="architecture-section__list">
              <li>Fine-tuning and training pipelines</li>
              <li>Fine-Tuned and Fine-Tuned + RAG live inference</li>
              <li>Automated grounding scores</li>
              <li>Server-side evaluation storage</li>
              <li>Shared classroom database</li>
              <li>Authentication and user accounts</li>
            </ul>
          </div>
        </div>
      </section>
    </>
  );
}
