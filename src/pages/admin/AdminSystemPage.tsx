import { Callout } from '../../components/ui/Callout';
import { PageHeader } from '../../components/ui/PageHeader';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { getConfiguredApiBaseUrl } from '../../lib/adminApi';

/**
 * Architecture and infrastructure reference.
 *
 * This is the one place in the application where implementation detail is the
 * point. It used to be a top-level navigation item visible to every student.
 *
 * It describes what is deployed, so it has to move when the deployment does:
 * the version of this page that shipped with authentication still said
 * authentication was not implemented. The "Not yet implemented" section is
 * kept to what docs/remaining-work.md lists as genuinely deferred.
 */

const COURSE_TABLES: { table: string; holds: string }[] = [
  { table: 'courses', holds: 'Course metadata and syllabus status, including the chunk count recorded when the syllabus was indexed' },
  { table: 'starter_seed_generation', holds: 'State of the automatic starter-seed job, one row per course' },
  { table: 'seed_examples', holds: 'Example questions, their review state, evidence and edit history' },
  { table: 'evaluations', holds: 'Student ratings of the four answers, attributed to a participant id' },
  { table: 'course_models / course_model_versions', holds: 'The per-course model registry' },
  { table: 'model_requests', holds: 'The professor-facing "I want a model" lifecycle' },
  { table: 'training_runs', holds: 'The queue the cluster claims work from, and what each run reported' },
  { table: 'serving_sessions', holds: 'Whether a GPU is serving fine-tuned inference, and until when' },
];

const IDENTITY_TABLES: { table: string; holds: string }[] = [
  { table: 'users', holds: 'Professor and administrator accounts: email, display name, global role, scrypt password hash' },
  { table: 'participants', holds: 'Anonymous students: a random id bound to one course, and nothing identifying' },
  { table: 'auth_sessions', holds: 'Every live session, staff or participant, as a hashed token with expiry and last-seen time' },
  { table: 'course_memberships', holds: 'Which professors instruct which courses. Administrators need no rows here' },
  { table: 'invitations', holds: 'Reusable class codes, and single-use professor, admin and reset links stored as token hashes' },
  { table: 'admin_actions', holds: 'Append-only audit trail of privileged actions, including administrators reviewing examples' },
];

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
            The route guards in the browser only decide where to send someone who
            is not allowed somewhere; every request is authorized again by the
            backend.
          </p>
          <p>
            Every request goes to FastAPI under <code>/api/</code>, carries the
            session cookies, and — when it changes anything — the CSRF header.
            The browser never talks to PostgreSQL.
          </p>
        </div>
      </section>

      <section className="ui-stack">
        <SectionHeader
          title="Identity and access"
          description="Implemented and enforced by the backend on every route."
          divider
        />
        <div className="ui-prose">
          <ul>
            <li>
              <strong>Students are anonymous, course-specific participants.</strong>{' '}
              An instructor puts the join page and a six-character class code on
              the board; entering the code makes this browser a participant in
              that one course. No name, email or NetID is asked for or stored,
              and a student in two courses is two unrelated participants.
            </li>
            <li>
              <strong>Professors and administrators sign in with email and
              password.</strong> Accounts exist only through single-use
              invitation links from an administrator; the first administrator
              comes from a command-line bootstrap link. Password reset is an
              administrator-issued one-time link.
            </li>
            <li>
              <strong>Two separate sessions.</strong> The staff session and the
              participant session are different cookies backed by different rows
              in <code>auth_sessions</code>, so a professor can walk the student
              flow of their own course without signing out.
            </li>
            <li>
              <strong>Sessions live in PostgreSQL.</strong> A cookie holds a
              random token; the database holds its hash, the principal, an
              absolute deadline and a last-seen time for the idle rule. Sign-out,
              disabling an account and changing a password revoke rows and take
              effect on the next request.
            </li>
            <li>
              <strong>Authorization is enforced by the backend.</strong> Every
              route declares its guard — administrator, course staff, course
              access, participant — and the test suite fails if a route is
              mounted without one. A professor reaches exactly the courses in{' '}
              <code>course_memberships</code>, compared against the course in the
              request path on every call; an administrator reaches every course
              and the technical surface without any membership.
            </li>
            <li>
              <strong>Privileged actions are audited.</strong> Invitations,
              role and membership changes, disabling accounts, clearing research
              data and an administrator&apos;s example reviews each write a row to{' '}
              <code>admin_actions</code>, shown under Audit.
            </li>
          </ul>
        </div>
      </section>

      <section className="ui-stack">
        <SectionHeader
          title="Data model"
          description="PostgreSQL is the system of record for all application state. Every course-scoped table has course_id in its primary key."
          divider
        />
        <ul className="admin-rows" aria-label="Course tables">
          {COURSE_TABLES.map((row) => (
            <li key={row.table} className="admin-row">
              <span className="admin-row__label">
                <code>{row.table}</code>
              </span>
              <span className="admin-row__value">{row.holds}</span>
            </li>
          ))}
        </ul>
        <ul className="admin-rows" aria-label="Identity tables">
          {IDENTITY_TABLES.map((row) => (
            <li key={row.table} className="admin-row">
              <span className="admin-row__label">
                <code>{row.table}</code>
              </span>
              <span className="admin-row__value">{row.holds}</span>
            </li>
          ))}
        </ul>
        <ul className="admin-rows" aria-label="Storage locations">
          <li className="admin-row">
            <span className="admin-row__label">Syllabus text</span>
            <span className="admin-row__value">
              <code>backend/course_data/{'{courseId}'}/syllabus.txt</code> on the VM,
              beside the uploaded file
            </span>
          </li>
          <li className="admin-row">
            <span className="admin-row__label">Retrieval index</span>
            <span className="admin-row__value">
              <code>backend/data/indexes/{'{courseId}'}.json</code>, with the cached
              fact inventory beside it as <code>.facts.json</code>
            </span>
          </li>
          <li className="admin-row">
            <span className="admin-row__label">Training datasets</span>
            <span className="admin-row__value">
              <code>data/exports/{'{courseId}'}/</code>, fetched by the cluster over the
              worker token
            </span>
          </li>
          <li className="admin-row">
            <span className="admin-row__label">API base URL</span>
            <span className="admin-row__value">
              <code>{getConfiguredApiBaseUrl() ?? 'not configured'}</code>
            </span>
          </li>
        </ul>
        <div className="ui-prose">
          <p>
            The earlier Firebase Realtime Database is gone. Nothing live reads or
            writes a <code>courses/{'{courseId}'}/…</code> path; the snapshot
            reader and one-time importer are kept only so the migration can be
            audited.
          </p>
        </div>
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
          <p>
            Model requests, the training queue, job status, automatic
            registration, and publication are implemented. Training and
            fine-tuned inference execute on Tillicum, which this application
            reaches through an authenticated queue API rather than by running
            anything itself.
          </p>
        </div>
      </section>

      <section className="ui-stack">
        <SectionHeader
          title="Not yet implemented"
          description="Deliberately deferred. Everything not listed here is in production."
          divider
        />
        <div className="ui-prose">
          <ul>
            <li>
              <strong>Self-service password reset and UW SSO.</strong> There is
              no mail path, so a forgotten staff password is an
              administrator-issued link; SSO would replace the password check
              behind the same sessions.
            </li>
            <li>
              <strong>Course deletion and participant erasure in the UI.</strong>{' '}
              The schema cascades correctly; neither is exposed in the
              interface, and deletion does not touch filesystem artifacts or the
              cluster.
            </li>
            <li>
              <strong>Reproducible evaluation provenance.</strong> An evaluation
              records which approach a student preferred, but not the answers as
              generated, the retrieved passages, or the model versions that
              produced them.
            </li>
            <li>
              <strong>Retention, redaction, and alerting.</strong> Nothing
              expires, nothing redacts personal information typed into a
              question or comment, and a failed background job is discoverable
              by looking rather than by being told.
            </li>
            <li>
              <strong>Shared throttling and a separate developer role.</strong>{' '}
              Join-code and login limits live in one process&apos;s memory, and
              the single privileged role covers both administration and
              development.
            </li>
            <li>
              <strong>Horizontal scaling.</strong> Syllabus artifacts and
              indexes are local disk, so the backend runs as one instance.
            </li>
          </ul>
        </div>
      </section>
    </div>
  );
}
