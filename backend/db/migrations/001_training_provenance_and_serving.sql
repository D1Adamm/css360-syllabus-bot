-- Training provenance, idempotent model registration, and serving sessions.
--
-- Idempotent: every statement is ADD COLUMN IF NOT EXISTS / CREATE ... IF NOT
-- EXISTS, so running it twice is a no-op and running it against a database
-- created from the current schema.sql is also a no-op. Nothing is dropped,
-- nothing is rewritten, and no existing row changes value: the three added
-- columns are nullable and the added table is new.
--
-- Why each piece exists
-- ---------------------
-- training_runs.completion
--   Everything the cluster knows when a job ends and the application does not:
--   which optimizer steps actually ran, the loss values, the measured GPU
--   hours, the git commit the code was at, the dataset digests the job trained
--   on, and where the artifact was written. JSONB rather than columns because
--   it is an operator-facing record whose shape is still moving, it is never
--   queried by field, and the alternative is a migration every time the runtime
--   report grows a key.
--
-- course_model_versions.run_id + uq_course_model_versions_run
--   The idempotency guarantee for automatic registration. A completion callback
--   that is delivered twice — a retry after a timeout, a worker re-run after a
--   dropped session — must not produce v2 for the same training run. The
--   partial unique index makes that impossible in the database rather than only
--   in the code path that happens to check first; rows registered before this
--   migration have a NULL run_id and are excluded from the constraint, so
--   existing history is untouched.
--
-- course_model_versions.provenance
--   The traceable facts about one artifact: base model, dataset reference and
--   checksums, approved/train/validation counts, resolved training config,
--   optimizer-step accounting, Slurm job id, git commit. Stored on the version
--   because that is the row an operator is looking at when they ask what a
--   model was made from, and it must survive the run row being read past.
--
-- serving_sessions
--   The one thing neither table could answer: whether a GPU inference job is
--   currently up, until when, and on which node. Deployment status on a model
--   version says whether a model is meant to be served; this says whether
--   anything is actually serving right now, which is a property of a Slurm job
--   with a wall clock on it rather than of any one course.

ALTER TABLE training_runs
    ADD COLUMN IF NOT EXISTS completion JSONB;

ALTER TABLE course_model_versions
    ADD COLUMN IF NOT EXISTS run_id TEXT;

ALTER TABLE course_model_versions
    ADD COLUMN IF NOT EXISTS provenance JSONB;

-- One registered version per training run, per course. Partial so that the
-- pre-existing versions with no recorded run are all still legal.
CREATE UNIQUE INDEX IF NOT EXISTS uq_course_model_versions_run
    ON course_model_versions(course_id, run_id)
    WHERE run_id IS NOT NULL;


CREATE TABLE IF NOT EXISTS serving_sessions (
    session_id TEXT PRIMARY KEY,

    -- Slurm job id of the inference allocation. Digits only, validated in the
    -- application; stored as TEXT because it is an identifier, not a number.
    job_id TEXT NOT NULL,

    -- Compute node and port the service is listening on. The node changes every
    -- job, which is why it is recorded rather than configured.
    node TEXT NOT NULL,
    port INTEGER NOT NULL,

    -- starting | ready | stopped | expired
    state TEXT NOT NULL,

    started_at TIMESTAMPTZ NOT NULL,
    -- When the allocation's wall clock runs out. A session past this is over
    -- whether or not anything reported it, which is what makes an operator's
    -- dropped login session harmless.
    expires_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,

    -- Courses served and their adapter versions, plus anything else the start
    -- script recorded. Operator-facing detail, never queried by field.
    detail JSONB,

    CONSTRAINT serving_session_port_range
        CHECK (port > 0 AND port < 65536)
);

CREATE INDEX IF NOT EXISTS idx_serving_sessions_expires
    ON serving_sessions(expires_at DESC);
