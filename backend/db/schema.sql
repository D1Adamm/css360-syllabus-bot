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

    PRIMARY KEY (course_id, version),

    CONSTRAINT model_training_count_nonnegative
        CHECK (training_example_count >= 0)
);


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
