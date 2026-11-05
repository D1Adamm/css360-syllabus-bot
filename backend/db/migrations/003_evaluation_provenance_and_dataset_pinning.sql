-- 003: evaluation provenance and dataset pinning.
--
-- Audit findings B8/B9 (2026-10-27): a rating recorded which approach won but
-- not the answers it was choosing between, and nothing stopped one comparison
-- being rated twice. B10: a queued run named its dataset by directory, so the
-- files could be re-prepared underneath it.
--
-- Additive and idempotent, like 001 and 002. No existing row changes value.

-- The four answers the student saw when rating, as the browser held them:
-- text, error, sources, and which model, version and engine produced each.
-- Rows written before this column stay NULL; they recorded a rating without
-- its answers and nothing here invents them.
ALTER TABLE evaluations ADD COLUMN IF NOT EXISTS responses JSONB;

-- One participant rates one comparison run once. The write enforces it; this
-- index makes the check cheap. Not UNIQUE: rows written before the rule may
-- already repeat, and a unique index would refuse to build over them.
CREATE INDEX IF NOT EXISTS idx_evaluations_participant_run
    ON evaluations(course_id, participant_id, run_id);

-- The digest of the prepared dataset at the moment the run was queued. The
-- worker is refused the dataset when the files on disk no longer match it.
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS dataset_sha256 TEXT;
