import { Callout } from '../../components/ui/Callout';
import { PageHeader } from '../../components/ui/PageHeader';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { getConfiguredApiBaseUrl } from '../../lib/adminApi';

/**
 * Architecture and infrastructure reference.
 *
 * This is the one place in the application where implementation detail is the
 * point. It used to be a top-level navigation item visible to every student.
 */
export function AdminSystemPage() {
  return (
    <div className="ui-stack ui-stack--loose">
      <PageHeader
        title="Architecture"
        eyebrow="Admin"
        description="How Syllabus Model Lab is put together and which model paths are live."
      />

      <Callout tone="info" title="Internal reference">
        Everything on this page is implementation detail. It is intentionally
        not reachable from the student or professor experience.
      </Callout>

      <section className="ui-stack">
        <SectionHeader title="Frontend" divider />
        <div className="ui-prose">
          <p>
            React 19 and TypeScript, bundled by Vite and served by Nginx on the
            UWB VM. React Router serves three role-scoped trees —{' '}
            <code>/student</code>, <code>/professor</code> and <code>/admin</code> —
            plus redirects from the earlier <code>/course/:courseId/*</code> URLs.
          </p>
          <p>
            Course metadata, example questions, and evaluations live in
            PostgreSQL, every table scoped by <code>courseId</code>, reached
            through FastAPI under <code>/api/</code>.
          </p>
        </div>
      </section>

      <section className="ui-stack">
        <SectionHeader title="Data model" divider />
        <ul className="admin-rows" aria-label="Storage locations">
          <li className="admin-row">
            <span className="admin-row__label">Course records</span>
            <span className="admin-row__value">
              <code>courses/{'{courseId}'}/metadata|seedExamples|evaluations</code>
            </span>
          </li>
          <li className="admin-row">
            <span className="admin-row__label">Syllabus text</span>
            <span className="admin-row__value">
              <code>backend/course_data/{'{courseId}'}/syllabus.txt</code>
            </span>
          </li>
          <li className="admin-row">
            <span className="admin-row__label">Retrieval index</span>
            <span className="admin-row__value">
              <code>backend/data/indexes/{'{courseId}'}.json</code>
            </span>
          </li>
          <li className="admin-row">
            <span className="admin-row__label">API base URL</span>
            <span className="admin-row__value">
              <code>{getConfiguredApiBaseUrl() ?? 'not configured'}</code>
            </span>
          </li>
        </ul>
      </section>

      <section className="ui-stack">
        <SectionHeader title="Backend and models" divider />
        <div className="ui-prose">
          <p>
            A FastAPI service handles syllabus upload and extraction, chunking,
            embeddings, course-scoped retrieval, and generation. Three distinct
            model roles are involved and are worth keeping apart:
          </p>
          <ul>
            <li>
              <strong>Answering model</strong> — serves the Base and RAG paths
              through Ollama.
            </li>
            <li>
              <strong>Embedding model</strong> — <code>nomic-embed-text</code>,
              used only to build and query the retrieval index.
            </li>
            <li>
              <strong>Example-generation model</strong> — used offline to draft
              starter examples and extract the fact inventory. Configurable, and
              not necessarily the same model that answers questions.
            </li>
            <li>
              <strong>Fine-tuned inference service</strong> — a separate service
              named by <code>FINETUNED_SERVICE_URL</code>, serving the
              Fine-Tuned and Fine-Tuned + RAG paths.
            </li>
          </ul>
          <p>
            All four comparison paths are implemented. Base and RAG depend on
            the local model runtime; Fine-Tuned availability depends on the
            configured inference service, which may or may not be running at any
            given moment — see Overview for its live state. Base and RAG share
            one CPU-bound process and are therefore issued sequentially; the two
            fine-tuned paths run against the separate service and overlap with
            them.
          </p>
        </div>
      </section>

      <section className="ui-stack">
        <SectionHeader title="Not yet implemented" divider />
        <div className="ui-prose">
          <p>
            Authentication and access control; student enrolment and course join
            codes; retention and redaction policies; reproducible evaluation
            provenance — an evaluation records which approach a student preferred,
            but not the answers as generated or the model versions that produced
            them.
          </p>
          <p>
            Model requests, the training queue, job status, automatic
            registration, and publication are implemented. Training and
            fine-tuned inference execute on Tillicum, which this application
            reaches through an authenticated queue API rather than by running
            anything itself.
          </p>
          <p>
            Course records, examples, evaluations, the model registry, and the
            training queue are stored in PostgreSQL and reached only through the
            backend. Syllabus artifacts and retrieval indexes are written to
            local disk, under <code>backend/course_data/</code> and{' '}
            <code>backend/data/indexes/</code>.
          </p>
        </div>
      </section>
    </div>
  );
}
