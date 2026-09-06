CREATE TABLE IF NOT EXISTS courses (
    course_id TEXT PRIMARY KEY,

    name TEXT NOT NULL,
    title TEXT NOT NULL,
    term TEXT NOT NULL,
    instructor_name TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL,

    syllabus_status TEXT NOT NULL,
    syllabus_file_name TEXT,
    syllabus_type TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,

    CONSTRAINT courses_chunk_count_nonnegative
        CHECK (chunk_count >= 0)
);


CREATE TABLE IF NOT EXISTS starter_seed_generation (
    course_id TEXT PRIMARY KEY
        REFERENCES courses(course_id)
        ON DELETE CASCADE,

    status TEXT,
    target_count INTEGER,
    final_count INTEGER,
    saved_count INTEGER,
    failed_to_save_count INTEGER,

    error TEXT,

    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Why a short run was short. Written by the automatic starter job and read
    -- by operators, not by the UI: a course whose syllabus only supports eleven
    -- examples and produced eleven is otherwise indistinguishable, by count
    -- alone, from one whose fact extractor was silently failing.
    achievable_ceiling INTEGER,
    limiting_factor TEXT,

    CONSTRAINT starter_ceiling_nonnegative
        CHECK (achievable_ceiling IS NULL OR achievable_ceiling >= 0),

    CONSTRAINT starter_target_nonnegative
        CHECK (target_count IS NULL OR target_count >= 0),

    CONSTRAINT starter_final_nonnegative
        CHECK (final_count IS NULL OR final_count >= 0),

    CONSTRAINT starter_saved_nonnegative
        CHECK (saved_count IS NULL OR saved_count >= 0),

    CONSTRAINT starter_failed_nonnegative
        CHECK (failed_to_save_count IS NULL OR failed_to_save_count >= 0)
);


CREATE TABLE IF NOT EXISTS seed_examples (
    seed_id TEXT NOT NULL,
    course_id TEXT NOT NULL
        REFERENCES courses(course_id)
        ON DELETE CASCADE,

    instruction TEXT NOT NULL,
    response TEXT NOT NULL,
    category TEXT NOT NULL,
    source_section TEXT NOT NULL,

    difficulty TEXT NOT NULL,
    directly_answered BOOLEAN NOT NULL,
    origin TEXT NOT NULL,

    notes TEXT,
    created_at TIMESTAMPTZ,
    status TEXT,
    question_type TEXT,

    source_chunk_ids JSONB,

    validation JSONB,

    review_status TEXT,
    review_notes TEXT,
    reviewed_at TIMESTAMPTZ,

    fact_id TEXT,
    evidence_quote TEXT,

    normalized_question_key TEXT,

    original_question TEXT,
    original_answer TEXT,

    was_edited BOOLEAN NOT NULL DEFAULT FALSE,

    PRIMARY KEY (course_id, seed_id)
);


CREATE INDEX IF NOT EXISTS idx_seed_examples_course
    ON seed_examples(course_id);

CREATE INDEX IF NOT EXISTS idx_seed_examples_review_status
    ON seed_examples(course_id, review_status);

CREATE INDEX IF NOT EXISTS idx_seed_examples_origin
    ON seed_examples(course_id, origin);

CREATE INDEX IF NOT EXISTS idx_seed_examples_normalized_question
    ON seed_examples(course_id, normalized_question_key);


CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id TEXT NOT NULL,
    course_id TEXT NOT NULL
        REFERENCES courses(course_id)
        ON DELETE CASCADE,

    comparison_id TEXT NOT NULL,

    most_accurate TEXT NOT NULL,
    most_helpful TEXT NOT NULL,
    most_concise TEXT NOT NULL,
    best_grounded TEXT NOT NULL,
    preferred_model TEXT NOT NULL,

    hallucination_flags JSONB NOT NULL DEFAULT '[]'::jsonb,

    comment TEXT,
    created_at TIMESTAMPTZ NOT NULL,

    run_id TEXT,
    question_text TEXT,

    PRIMARY KEY (course_id, evaluation_id)
);


CREATE INDEX IF NOT EXISTS idx_evaluations_course_created
    ON evaluations(course_id, created_at DESC);


CREATE TABLE IF NOT EXISTS course_models (
    course_id TEXT PRIMARY KEY
        REFERENCES courses(course_id)
        ON DELETE CASCADE,

    current_version TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS course_model_versions (
    course_id TEXT NOT NULL
        REFERENCES courses(course_id)
        ON DELETE CASCADE,

    version TEXT NOT NULL,

    base_model TEXT NOT NULL,
    training_example_count INTEGER NOT NULL,

    status TEXT NOT NULL,
    deployment TEXT NOT NULL,

    artifact_ref TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ,
    notes TEXT,

    -- The training run this artifact came from, when one is known. NULL for
    -- versions registered by hand before automatic registration existed.
    run_id TEXT,

    -- Traceability for one artifact: base model, dataset reference and
    -- checksums, example counts, resolved training configuration, optimizer
    -- step accounting, Slurm job id, git commit. See
    -- backend/db/migrations/001_training_provenance_and_serving.sql.
    provenance JSONB,

    PRIMARY KEY (course_id, version),

    CONSTRAINT model_training_count_nonnegative
        CHECK (training_example_count >= 0)
);


-- At most one registered version per training run. This is what makes a
-- repeated completion callback idempotent rather than a source of v2, v3, …
CREATE UNIQUE INDEX IF NOT EXISTS uq_course_model_versions_run
    ON course_model_versions(course_id, run_id)
    WHERE run_id IS NOT NULL;


CREATE TABLE IF NOT EXISTS model_requests (
    course_id TEXT PRIMARY KEY
        REFERENCES courses(course_id)
        ON DELETE CASCADE,

    status TEXT NOT NULL,

    requested_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,

    approved_example_count INTEGER NOT NULL,

    failure_message TEXT,

    preparation JSONB,
    preparation_error TEXT,

    training JSONB,
    launch_error TEXT,

    current_run_id TEXT,

    CONSTRAINT request_approved_count_nonnegative
        CHECK (approved_example_count >= 0)
);


CREATE TABLE IF NOT EXISTS training_runs (
    run_id TEXT NOT NULL,
    course_id TEXT NOT NULL
        REFERENCES courses(course_id)
        ON DELETE CASCADE,

    mode TEXT NOT NULL,
    state TEXT NOT NULL,

    enqueued_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,

    dataset_ref TEXT NOT NULL,

    approved_example_count INTEGER NOT NULL DEFAULT 0,
    train_examples INTEGER NOT NULL DEFAULT 0,
    validation_examples INTEGER NOT NULL DEFAULT 0,

    attempt INTEGER NOT NULL DEFAULT 0,

    job_id TEXT,

    claim_owner TEXT,
    claim_claimed_at TIMESTAMPTZ,
    claim_expires_at TIMESTAMPTZ,

    error TEXT,

    -- What the cluster reported when the job ended: optimizer steps completed
    -- against intended, losses, measured GPU hours, git commit, dataset
    -- digests, artifact location, failure stage. Operator-facing; never queried
    -- by field, which is why it stays whole in JSONB.
    completion JSONB,

    PRIMARY KEY (course_id, run_id),

    CONSTRAINT training_counts_nonnegative CHECK (
        approved_example_count >= 0
        AND train_examples >= 0
        AND validation_examples >= 0
        AND attempt >= 0
    )
);


CREATE INDEX IF NOT EXISTS idx_training_runs_course_state
    ON training_runs(course_id, state);

CREATE INDEX IF NOT EXISTS idx_training_runs_enqueued
    ON training_runs(course_id, enqueued_at);


CREATE TABLE IF NOT EXISTS serving_sessions (
    session_id TEXT PRIMARY KEY,

    job_id TEXT NOT NULL,

    node TEXT NOT NULL,
    port INTEGER NOT NULL,

    -- starting | ready | stopped | expired
    state TEXT NOT NULL,

    started_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,

    detail JSONB,

    CONSTRAINT serving_session_port_range
        CHECK (port > 0 AND port < 65536)
);


CREATE INDEX IF NOT EXISTS idx_serving_sessions_expires
    ON serving_sessions(expires_at DESC);


-- ---------------------------------------------------------------------------
-- Authentication, authorization, and pseudonymous participants.
--
-- Mirrors backend/db/migrations/002_auth_identity.sql, which carries the
-- rationale for every table and column. The three attribution columns are
-- added with ALTER TABLE at the end rather than inline because they reference
-- tables created below the ones they belong to; the result is identical.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY,

    -- Stored lowercased; the unique index below is on lower(email) as well so
    -- a differently-cased duplicate cannot slip in through another writer.
    email TEXT NOT NULL,
    display_name TEXT NOT NULL,

    -- admin | professor. Global role only; course access is a membership.
    role TEXT NOT NULL,

    -- Self-describing scrypt hash: scrypt$<log2 n>$<r>$<p>$<salt>$<key>.
    password_hash TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL,
    disabled_at TIMESTAMPTZ,
    last_login_at TIMESTAMPTZ,

    -- The invitation that created this account. NULL only for an account
    -- created by a path that had no invitation, which today is none.
    created_via_invitation_id UUID,

    CONSTRAINT users_role_known
        CHECK (role IN ('admin', 'professor'))
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email
    ON users (lower(email));


CREATE TABLE IF NOT EXISTS course_memberships (
    course_id TEXT NOT NULL
        REFERENCES courses(course_id)
        ON DELETE CASCADE,
    user_id UUID NOT NULL
        REFERENCES users(user_id)
        ON DELETE CASCADE,

    -- Only `instructor` today. A column rather than an implicit meaning so a
    -- second membership role can be added without a migration of intent.
    membership_role TEXT NOT NULL DEFAULT 'instructor',

    granted_by UUID
        REFERENCES users(user_id)
        ON DELETE SET NULL,
    granted_at TIMESTAMPTZ NOT NULL,

    PRIMARY KEY (course_id, user_id),

    CONSTRAINT course_memberships_role_known
        CHECK (membership_role IN ('instructor'))
);

CREATE INDEX IF NOT EXISTS idx_course_memberships_user
    ON course_memberships(user_id);


CREATE TABLE IF NOT EXISTS invitations (
    invitation_id UUID PRIMARY KEY,

    -- student | professor | admin | reset
    kind TEXT NOT NULL,

    -- Student invitations: the classroom code, normalised to the code
    -- alphabet (upper case, no ambiguous characters). Unique across every
    -- invitation ever issued so a lookup by code is never ambiguous.
    code TEXT,

    -- Privileged invitations: SHA-256 of the one-time token. The token itself
    -- is never stored.
    token_hash TEXT,

    -- Student invitations: the one course this code admits to.
    course_id TEXT
        REFERENCES courses(course_id)
        ON DELETE CASCADE,

    -- Reset invitations: the account whose password may be set.
    target_user_id UUID
        REFERENCES users(user_id)
        ON DELETE CASCADE,

    label TEXT,

    -- NULL only for the bootstrap admin invitation minted from the command
    -- line before any administrator exists.
    created_by UUID
        REFERENCES users(user_id)
        ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL,

    -- NULL means the invitation does not expire on its own. Student codes
    -- default to this; privileged invitations are always given a deadline.
    expires_at TIMESTAMPTZ,

    -- NULL means unlimited. Student codes are reusable by a whole class;
    -- privileged invitations are always 1.
    max_uses INTEGER,
    use_count INTEGER NOT NULL DEFAULT 0,

    revoked_at TIMESTAMPTZ,
    revoked_by UUID
        REFERENCES users(user_id)
        ON DELETE SET NULL,

    -- Privileged invitations: who accepted it, and when. A student code is
    -- accepted many times and records its redemptions in participants instead.
    accepted_at TIMESTAMPTZ,
    accepted_by_user_id UUID
        REFERENCES users(user_id)
        ON DELETE SET NULL,

    CONSTRAINT invitations_kind_known
        CHECK (kind IN ('student', 'professor', 'admin', 'reset')),

    -- A student code names a course and carries a code; every other kind
    -- carries a token hash. Nothing carries both.
    CONSTRAINT invitations_kind_shape CHECK (
        (kind = 'student' AND code IS NOT NULL AND course_id IS NOT NULL AND token_hash IS NULL)
        OR (kind <> 'student' AND token_hash IS NOT NULL AND code IS NULL)
    ),

    CONSTRAINT invitations_reset_has_target
        CHECK (kind <> 'reset' OR target_user_id IS NOT NULL),

    CONSTRAINT invitations_uses_nonnegative
        CHECK (use_count >= 0 AND (max_uses IS NULL OR max_uses > 0))
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_invitations_code
    ON invitations(code)
    WHERE code IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_invitations_token_hash
    ON invitations(token_hash)
    WHERE token_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_invitations_course
    ON invitations(course_id);

CREATE INDEX IF NOT EXISTS idx_invitations_kind_created
    ON invitations(kind, created_at DESC);


CREATE TABLE IF NOT EXISTS invitation_course_grants (
    invitation_id UUID NOT NULL
        REFERENCES invitations(invitation_id)
        ON DELETE CASCADE,
    course_id TEXT NOT NULL
        REFERENCES courses(course_id)
        ON DELETE CASCADE,

    PRIMARY KEY (invitation_id, course_id)
);


CREATE TABLE IF NOT EXISTS participants (
    -- The only student identifier in the system. Random, and meaningful to
    -- nothing outside this database.
    participant_id UUID PRIMARY KEY,

    course_id TEXT NOT NULL
        REFERENCES courses(course_id)
        ON DELETE CASCADE,

    -- Which classroom code admitted this participant. Kept so a revoked code's
    -- participants can be counted; SET NULL so revoking or deleting the code
    -- does not delete the research data behind it.
    invitation_id UUID
        REFERENCES invitations(invitation_id)
        ON DELETE SET NULL,

    created_at TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_participants_course
    ON participants(course_id);


CREATE TABLE IF NOT EXISTS auth_sessions (
    session_id UUID PRIMARY KEY,

    -- SHA-256 of the cookie value. The value itself is never stored.
    token_hash TEXT NOT NULL,

    -- user | participant
    principal_kind TEXT NOT NULL,
    user_id UUID
        REFERENCES users(user_id)
        ON DELETE CASCADE,
    participant_id UUID
        REFERENCES participants(participant_id)
        ON DELETE CASCADE,

    created_at TIMESTAMPTZ NOT NULL,
    -- Absolute deadline. Idle timeouts are applied on top of this from
    -- last_seen_at, in the application, so they can be tuned without a
    -- migration.
    expires_at TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,

    CONSTRAINT auth_sessions_kind_known
        CHECK (principal_kind IN ('user', 'participant')),

    CONSTRAINT auth_sessions_exactly_one_principal CHECK (
        (principal_kind = 'user' AND user_id IS NOT NULL AND participant_id IS NULL)
        OR (principal_kind = 'participant' AND participant_id IS NOT NULL AND user_id IS NULL)
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_auth_sessions_token_hash
    ON auth_sessions(token_hash);

CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires
    ON auth_sessions(expires_at);

CREATE INDEX IF NOT EXISTS idx_auth_sessions_user
    ON auth_sessions(user_id);

CREATE INDEX IF NOT EXISTS idx_auth_sessions_participant
    ON auth_sessions(participant_id);


CREATE TABLE IF NOT EXISTS admin_actions (
    action_id BIGSERIAL PRIMARY KEY,

    -- NULL when the actor was the bootstrap script rather than a signed-in
    -- administrator. SET NULL so the record outlives the account.
    actor_user_id UUID
        REFERENCES users(user_id)
        ON DELETE SET NULL,
    actor_role TEXT,

    -- Dotted verb, e.g. invitation.create, membership.add, user.disable.
    action TEXT NOT NULL,

    target_kind TEXT,
    target_id TEXT,

    -- Not a foreign key on purpose: an audit row must survive the course.
    course_id TEXT,

    -- Structured context. Never a credential, token, or password.
    detail JSONB,

    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_admin_actions_created
    ON admin_actions(created_at DESC);


-- Attribution on existing research data. Nullable: every existing row keeps
-- NULL, and nothing backfills an identity that was never recorded.

ALTER TABLE evaluations
    ADD COLUMN IF NOT EXISTS participant_id UUID
        REFERENCES participants(participant_id)
        ON DELETE SET NULL;

ALTER TABLE seed_examples
    ADD COLUMN IF NOT EXISTS participant_id UUID
        REFERENCES participants(participant_id)
        ON DELETE SET NULL;

ALTER TABLE courses
    ADD COLUMN IF NOT EXISTS created_by UUID
        REFERENCES users(user_id)
        ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_evaluations_participant
    ON evaluations(course_id, participant_id);

CREATE INDEX IF NOT EXISTS idx_seed_examples_participant
    ON seed_examples(course_id, participant_id);
