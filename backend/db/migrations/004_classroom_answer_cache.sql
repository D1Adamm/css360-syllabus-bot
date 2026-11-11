-- 004: classroom answer cache and classroom preparation.
--
-- Three new tables; no existing table or row changes. Additive and
-- idempotent, like 001-003.
--
-- Staff choose which of a course's existing questions to prepare before class
-- (`classroom_questions`). Each of the four approaches' real answers to a
-- chosen question is stored once per configuration (`answer_cache`), and one
-- row per course records how the last preparation run ended
-- (`classroom_preparation`). See docs/classroom-preparation.md.

-- Which course questions staff selected for classroom preparation. Separate
-- from the training review state on purpose: selecting a question here does
-- not approve it for training, and approving one there does not select it.
-- The question is referenced, not copied, so an edited question is prepared
-- under its current wording and a deleted one stops being prepared.
CREATE TABLE IF NOT EXISTS classroom_questions (
    course_id TEXT NOT NULL
        REFERENCES courses(course_id)
        ON DELETE CASCADE,
    seed_id TEXT NOT NULL,

    selected_at TIMESTAMPTZ NOT NULL,
    selected_by UUID
        REFERENCES users(user_id)
        ON DELETE SET NULL,

    PRIMARY KEY (course_id, seed_id),
    CONSTRAINT classroom_questions_seed
        FOREIGN KEY (course_id, seed_id)
        REFERENCES seed_examples(course_id, seed_id)
        ON DELETE CASCADE
);

-- One real answer from one approach to one question under one configuration.
--
-- `question_key` is the SHA-256 of the conservatively normalised question;
-- `fingerprint` is the SHA-256 of `identity`, which names everything that
-- decides the answer (model digests, runtime version, course model version,
-- syllabus index, prompt, decoding options, retrieval code, host CPU). A
-- change to any of them is a different fingerprint, so an old row is simply
-- no longer found; rows are kept so a rating can still point at the answer
-- it was given (`entry_id`).
--
-- No participant, user or session is recorded: a row says what a model
-- answered, never who asked.
CREATE TABLE IF NOT EXISTS answer_cache (
    entry_id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL
        REFERENCES courses(course_id)
        ON DELETE CASCADE,

    -- base | rag | fineTuned | fineTunedRag
    condition TEXT NOT NULL,
    question_key TEXT NOT NULL,
    fingerprint TEXT NOT NULL,

    -- The staff-selected question as it was asked when this row was written.
    question_text TEXT NOT NULL,

    -- The route's response body exactly as the model pipeline produced it.
    payload JSONB NOT NULL,
    -- What `fingerprint` was computed from.
    identity JSONB NOT NULL,
    -- How it was produced: timings, token counts, prompt hash, engine.
    provenance JSONB NOT NULL,

    -- prepared (the preparation run) | live (a student asked a selected question)
    source TEXT NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL,

    CONSTRAINT answer_cache_condition_known
        CHECK (condition IN ('base', 'rag', 'fineTuned', 'fineTunedRag')),
    CONSTRAINT answer_cache_source_known
        CHECK (source IN ('prepared', 'live'))
);

-- The lookup, and the rule that one configuration has one stored answer: the
-- first one written stays, and a second writer's insert is a no-op.
CREATE UNIQUE INDEX IF NOT EXISTS uq_answer_cache_key
    ON answer_cache(course_id, condition, question_key, fingerprint);

-- How a course's last preparation run ended. One row per course, overwritten
-- by the next run; progress itself is never stored here, it is counted from
-- `answer_cache`. `boot_id` names the backend process that ran it: a row
-- still marked running by a process that no longer exists was interrupted.
CREATE TABLE IF NOT EXISTS classroom_preparation (
    course_id TEXT PRIMARY KEY
        REFERENCES courses(course_id)
        ON DELETE CASCADE,

    -- running | stopping | completed | stopped | failed
    state TEXT NOT NULL,
    boot_id TEXT NOT NULL,

    started_at TIMESTAMPTZ NOT NULL,
    started_by UUID
        REFERENCES users(user_id)
        ON DELETE SET NULL,
    stop_requested_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,

    -- A short code and a plain sentence for why the run ended as it did.
    reason TEXT,
    message TEXT,

    generated_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,

    CONSTRAINT classroom_preparation_state_known
        CHECK (state IN ('running', 'stopping', 'completed', 'stopped', 'failed'))
);
