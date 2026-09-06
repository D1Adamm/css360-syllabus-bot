-- Authentication, authorization, and pseudonymous participant identity.
--
-- Idempotent: every statement is CREATE ... IF NOT EXISTS / ADD COLUMN IF NOT
-- EXISTS, so running it twice is a no-op and running it against a database
-- created from the current schema.sql is also a no-op. Nothing is dropped,
-- nothing is rewritten, and no existing row changes value: the three added
-- columns are nullable, and every existing evaluation and seed keeps a NULL
-- participant. Historical research rows stay anonymous in the way they always
-- were; only rows written after this migration carry an attribution.
--
-- Why each piece exists
-- ---------------------
-- users
--   Professors and administrators. The only people with accounts. Students
--   never appear here — see participants. `role` is the global role and says
--   nothing about which courses a professor may touch; that is
--   course_memberships. Email is the login name and is unique case-insensitively.
--
-- course_memberships
--   Which professors may act on which courses. A professor with no rows here
--   sees no courses. Multiple rows per course are co-instructors; multiple rows
--   per user are a professor teaching several courses. Administrators need no
--   rows: their access is global by role.
--
-- invitations
--   Every way into the system that is not a password. One table, four kinds:
--     student    reusable classroom code, bound to one course, redeemed by
--                anonymous participants. The short code is stored as typed
--                because it has to be shown on the professor's page again and
--                read out in a classroom; it grants only what a seat in that
--                class already grants, and a six-character code cannot be
--                protected by an unsalted hash anyway.
--     professor  single-use, expiring, creates a professor account and the
--                memberships listed in invitation_course_grants.
--     admin      single-use, short-lived, creates an administrator account.
--     reset      single-use, short-lived, lets an existing user set a new
--                password (there is no mail path for self-service reset).
--   The three privileged kinds store only a SHA-256 of a 256-bit random token;
--   the token itself is shown once and never persisted.
--
-- invitation_course_grants
--   The courses a professor invitation assigns on acceptance, with referential
--   integrity rather than a free-text list.
--
-- participants
--   The pseudonymous student identity: a random UUID bound to exactly one
--   course. Nothing identifying is stored — no name, email, NetID, address or
--   user agent. A student in two courses is two unrelated participants, which
--   is deliberate: activity is not linkable across courses.
--
-- auth_sessions
--   Server-side sessions for both kinds of principal, so a session can be
--   revoked, expired, and reasoned about from the database rather than trusted
--   from a cookie. The cookie holds a random token; only its SHA-256 is here.
--   The check constraint makes a session that is both a user and a participant,
--   or neither, impossible to store.
--
-- admin_actions
--   The audit trail for privileged actions: invitation lifecycle, role and
--   membership changes, account disabling, bulk deletion of research data.
--   `course_id` is deliberately not a foreign key so the record outlives the
--   course it names. Never holds a credential, token, or session secret.
--
-- evaluations.participant_id, seed_examples.participant_id
--   The research linkage. Nullable, because every row written before this
--   migration has none and nothing invents one. ON DELETE SET NULL so erasing
--   a participant anonymises its rows rather than destroying them.
--
-- courses.created_by
--   Provenance only. NULL for every course that exists today.


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
