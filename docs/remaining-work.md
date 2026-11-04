# Remaining work

What is genuinely unfinished. Everything not listed here is implemented and
covered by tests, as far as the repository-wide audit of 2026-10-27 found — if
you are looking for a feature and it is not below, look for it in the code
before building it. The audit's open findings are in §8; which of the four
workflows is verified and which is experimental is in
[workflows.md](workflows.md).

Ordered by what blocks real classroom use.

---

## 1. Identity: what is still open

Authentication, authorization, class codes and the audit trail are implemented
and covered (see [architecture.md](architecture.md#identity-and-access)). What
remains is the set of things deliberately left for after the Fall study:

- **Self-service password reset.** There is no mail path, so a forgotten
  password is an administrator-issued one-time link. A mail relay would allow
  self-service reset and, if wanted, magic-link sign-in as a second credential.
- **UW SSO** for staff is the natural long-term upgrade. The staff sign-in path
  is one function behind `/api/auth/login`; SSO would establish the same
  `auth_sessions` row without touching roles, memberships or guards.
- **Participant continuity.** A student's identity is the participant cookie in
  one browser; clearing it or changing devices starts a new participant. There
  is nothing to recover with, by design, and nothing links the two.
- **Throttling is per process.** Join-code and login failure limits live in
  memory in the single uvicorn process. Fine for one VM; a second instance
  would need shared state.
- **Developer and Admin are one role.** The permission model has one privileged
  role; splitting it means a new `users.role` value and a guard between
  `require_admin` and a stricter one.

## 2. Course deletion and participant erasure in the UI

The schema cascades correctly (deleting a course removes its codes,
participants, memberships and research rows; deleting a participant anonymises
its rows), but neither is exposed in the interface.

## 3. Research provenance and evaluation reproducibility

Training-side provenance is strong: every registered version records its dataset
checksums, resolved configuration, optimizer-step accounting, git commit, Slurm
job id, and measured GPU hours.

The **evaluation** side is weaker. An evaluation records which approach a student
preferred and the question text, but not the four answers as generated, the
retrieved passages, or the exact model versions that produced them. A finding
cannot currently be reproduced from stored data alone — the models may have moved
underneath it.

Worth having before results are published:

- Persist the generated answers and retrieved passages with the comparison run.
- Record the resolved model version for each approach at answer time.
- Export a stable, versioned research dataset rather than reading live tables.

## 4. Privacy, retention, and monitoring

- **No retention policy.** Nothing expires. Contributed questions, evaluations,
  uploaded syllabi, indexes, and training artifacts accumulate indefinitely.
- **No redaction.** A student can type personal information into a contributed
  question or an evaluation comment, and it is stored verbatim.
- **No structured application logging or alerting.** A failed starter-seed job,
  a stopped `aiswe-finetuned` unit, or an expired fallback serving session is
  discoverable by looking, not by being told.
- **Course deletion does not touch the filesystem artifacts or the cluster.**

`training/cleanup_training_outputs.sh` covers cluster disk only, is dry-run by
default, and never proposes a published adapter.

## 5. Backend scalability

Syllabus artifacts and embedding indexes are written to the VM's local disk. One
backend process owns them, so the backend cannot be run as more than one
instance without moving that storage. Not a problem at classroom scale; it is the
first thing to hit if this grows.

## 6. Fine-tuned inference without Tillicum

**Inference: done.** Fine-Tuned and Fine-Tuned + RAG are served on the UWB VM by the
`aiswe-finetuned` user unit (`training/inference_service/ollama_service.py`
against the VM's own Ollama, on `127.0.0.1:9001`, the same contract the GPU
service answers). It is installed, mapped, verified and switched to and from
the Tillicum fallback with tracked tooling (`scripts/aiswe_finetuned.sh`,
`scripts/install_finetuned_adapter.py`, `scripts/verify_finetuned_production.py`,
`scripts/finetuned_latency_probe.py`); see
[deployment.md](deployment.md#fine-tuned-inference-on-the-vm). Ordinary
student use needs no Tillicum allocation, SSH tunnel, Duo prompt, laptop,
`caffeinate` or open terminal. Every served course runs this way: CSS 350,
CSS 360, and the Fall courses CSS 360D and CSS 360E
([verification-history.md](verification-history.md#css-360d-and-css-360e-on-the-vm)).
The Tillicum GPU service remains only as the emergency fallback and reference
implementation.

**Training: Tillicum is still in the loop, and that is not the end state.**
Removing Tillicum from training is future architecture work, after the Fall
classroom evaluation. What still depends on it today:

- **Queued QLoRA training** runs on Tillicum GPUs (`run_training_queue.sh`,
  started by a person after a Duo login). The CPU path (`train_qlora.py --cpu`)
  exists but is launched and registered by hand rather than through the queue;
  it is **experimental** ([workflows.md §4](workflows.md#4-vm-cpu-training-experimental)):
  one test adapter so far, no recorded cycle to an activated version.
Publication no longer needs Tillicum: an administrator activates a version
from Admin → Models once the VM serves it, and only an activated version
answers — finishing or registering training never moves traffic on its own.
`promote_qlora_adapter.sh` still records publications too, as the legacy
path, checked against the VM's fine-tuned service exactly as activation is.

Still by hand, deliberately, wherever training runs: deciding that a trained
version should be served (install and map it on the VM, then activate it).

---

## 7. Serving and concurrency (root cause found and fixed 2026-10-01)

The in-class failures of 2026-09-30 and 2026-10-01 have one main cause. A
course model is `FROM llama3.2:3b` + `ADAPTER`, so it shares the base model's
weights file, and one Ollama keeps at most one runner per weights file. With
Base/RAG and the fine-tuned conditions each behind their own lock against the
same Ollama, nearly every request swapped the model. Unbounded waits then
turned that into late 503s. The full account, the fix (a second Ollama for the
course models, and one bounded queue for all generation), the deployment
steps and the load test are in [classroom-capacity.md](classroom-capacity.md).

Still open:

- **Measure the VM** with `scripts/classroom_load_test.py` after deploying
  (still open on 2026-10-31: no recorded result),
  then decide `num_thread` 4 vs 8 and whether `GENERATION_MAX_CONCURRENCY=2`
  with `OLLAMA_NUM_PARALLEL=2` helps. Defaults stay at one generation at a
  time until measured.
- **Raw capacity.** 8 vCPUs cannot finish thirty simultaneous four-condition
  comparisons within a few minutes. The queue makes the limit orderly and
  visible, but it does not remove it.
- **Two courses in class at once** share the second Ollama, so their models
  swap with each other in the same way. One more server per concurrently
  taught course would fix that if it ever happens.
- **A running starter call** holds the generation slot until it finishes.
  Classroom requests go first at every hand-off, but not mid-call.
- **Refresh recovery** for a comparison in progress, and how often
  `classroom-concise-v1` answers reach the 128-token cap
  (`docs/css360-model-evolution.md` §13).

## 8. Open findings from the 2026-10-27 audit

A read-only, repository-wide audit on 2026-10-27 (commit `211f6e9`) rated
three findings High and the rest Medium or Low. The three High ones are closed:

- **B1** (the legacy publication route could point a course at a version the
  VM did not serve) and **B2** (the worker's repair step could move a finished
  run back to `submitted`) were fixed in `ef7b7c7`, deployed as `568a480` on
  2026-10-31.
- **B3** (the backend unit and the Nginx site were never checked against the
  code's assumptions) was closed by the 2026-10-31 inspection
  ([verification-history.md](verification-history.md#production-vm-configuration-2026-10-31));
  reference copies are in [templates/](templates/README.md). The inspection also
  found and `ef7b7c7` fixed the emergency fallback scripts' final check, which
  had probed an administrator-only route since 2026-09-04.

Fixed in code on 2026-11-03 (branch `fix/audit-medium-findings`; not yet
deployed — the deployment carries migration 003, see
[deployment.md](deployment.md#database-migrations-if-any)):

- **B7** The backend classifies every fine-tuned, Ollama and model-resolution
  failure (`app/error_classification.py`: `{"detail": {"code", "message"}}`),
  `GET /api/admin/model-readiness` judges every registered course against what
  the service serves, the admin Overview shows it per course, and a background
  task (`FINETUNED_READINESS_CHECK_SECONDS`, default 300 s) logs a WARNING to
  the journal when a problem appears and INFO when it clears. Still open from
  B7: the verify script SKIPs the backend checks without credentials.
- **B5** The browser maps each code to its own sentence for students and
  professors (`src/lib/errorMessages.ts`): "no published course model yet",
  "service not answering", "took too long" and "syllabus not prepared" no
  longer read alike, and each says what still works. Admin keeps the raw text.
- **B4** The Professor Model page and Course overview describe the activated
  version (`deployment = online`, `status = ready`, the backend's own rule),
  falling back to the newest only when nothing is activated, with one line
  about a newer version that is trained-and-waiting, training or failed.
- **B8** Each rating stores a snapshot of the four answers with the model,
  version and engine that produced them (`evaluations.responses`), and the
  rated run is marked rated in the browser. Retrieved passages are still only
  the source labels the student saw, not the chunk texts.
- **B6** `edited` is now hidden from students on the same rule the export
  uses (`approved` only); the review pages label it "Edited · not yet
  approved", say how many are waiting, and offer "Save and approve".
- **B9** The create-evaluation route validates every participant-writable
  field (approach keys, lengths, the snapshot shape), stamps `created_at`
  itself and refuses a second rating of the same run (`409
  evaluation_already_recorded`).
- **B10** Every run — queued from either enqueue route or created by Retry —
  records the SHA-256 of the prepared files; the worker is refused the dataset
  of a run whose files changed (`409 dataset_changed`) and reports the run as
  refused (`POST …/runs/{run_id}/refused`, 2026-11-04): the run ends `failed`
  with the refusal as its reason and is never claimed again, while the request
  stays `preparing` with the reason on `launchError` (no job ran, so the
  professor is not told one failed). Retry, or Queue training, then queues a
  replacement pinned to the current files. Transient refusals — data not yet
  prepared, a file that did not transfer — still release the run for the next
  cycle. The export and split routes refuse while a worker holds the files
  (`claimed`, `submitted`, `training`: `409 training_run_active`), and the
  admin Training page hides Re-prepare in that state and shows the digests.
  Tillicum must pull before the next run: a worker on older code still
  releases a refused run and re-claims it every cycle.
- **B14** Training and serving render prompts from one module
  (`training/llama3_prompt.py`): `train_qlora.py` no longer calls
  `apply_chat_template`, refuses a tokenizer that does not know the Llama 3
  special tokens, counts every record's tokens before training and fails (or,
  with `--on-oversized drop`, drops and reports) anything over the window
  instead of truncating it. Adapters trained before this were trained on the
  older rendering; retraining is needed to benefit, as the audit said.

Open, Medium unless marked. Ordered as the audit's action plan ordered them:

- **B13** A failed session check (5xx, timeout) is treated as "signed out".
- **B11** The course → Ollama-tag mapping has no cross-check; a typo serves
  another course's adapter with every check green.
- **B12** The launcher's "existing active job" short-circuit can attribute
  another run's job to a new run (likely).
- **B15** A job that dies without running its trap strands the run at
  `submitted`; no `sacct` reconciliation.
- **B18** The automatic starter-seed job defaults on in code (`.env` must say
  `false`) and cannot recover from a restart mid-run.
- **B16** The GPU failback watchdog judges `/health` only, not generation
  (likely).
- **B17** The per-account login throttle lets anyone lock a professor out.
- **B19** A professor on the student pages without joining dead-ends
  (Medium-Low).
- **B20** The second Ollama is wired by an env-file edit the installer does not
  make, and the verifier never checks which Ollama answers (Medium-Low; the
  example env file now carries the production value).

Low findings worth knowing about: a completion refused with a status other
than 404/409 is retried on every flush forever (L1); the Admin Training page
never polls (L2); `/styleguide` is routed in production (L3); a comparison id
can hold a queue position for 600 s (L4); uploads embed outside the generation
queue (L5); hardcoded CSS 360 defaults in the Slurm scripts (L6);
`backend/requirements.txt` is unpinned (L11; the VM runs FastAPI 0.141.1,
uvicorn 0.52.0); dead launch paths are still mounted (L12); stale comments say
`/api/db` and `/api/fine-tuned/health` need no credential (L13); the Admin
System page still says all four conditions share one Ollama (L15).

Unexposed but implemented: own-password change (`POST /api/auth/password`),
single and bulk evaluation deletion. HTTP/2 on the Nginx TLS listener remains
the one recommended infrastructure change not yet applied.

## Known issues

- **Compare under classroom load** — §7. Root cause found and fixed in code
  2026-10-01; the second Ollama and the VM load test are deployment steps
  ([classroom-capacity.md](classroom-capacity.md)).
- **HTTP/2 is not enabled** on the Nginx TLS listener (verified 2026-10-31), so
  a browser has six connections per origin and a Compare run in one tab can
  starve another tab's requests; the client-side timeout and retry on the
  session check is the mitigation. `http2 on;` is a one-line, reloadable Nginx
  change ([templates/nginx-aiswe.conf](templates/nginx-aiswe.conf)).
- **The adapter installer needs two explicit flags for newer courses**, by
  design: `--tag` for course ids outside the legacy `css-360-…` form (a
  derived tag could collide across terms), and, on the VM,
  `--base-model-path <local snapshot>`, because `--base-model-id` alone makes
  the converter contact Hugging Face. See
  [deployment.md](deployment.md#installing-a-completed-adapter).

---

## Deliberately out of scope

Recorded so nobody re-proposes them as oversights.

- **Automating UW two-factor.** Not a gap. It will not be automated, stored, or
  worked around.
- **Training from the browser.** A web request cannot complete an interactive
  cluster login. The queue exists precisely so it does not have to: the browser
  writes a run and stops.
- **Automatic promotion of a newly trained model.** Training success means a
  usable artifact exists. Deciding to serve it is a person's judgement, and
  keeping the two separate is what stops a bad run replacing a working model.
- **Automatic retries of failed runs.** Retry is an explicit admin action. An
  automatic loop would burn GPU allocation on a systematically failing run.
- **Automatic node exclusion.** A node failing its GPU preflight today is usually
  repaired within days; excluding it permanently keeps the scheduler off healthy
  hardware. `--exclude-node` exists for the operator to use temporarily.
- **Returning to Firebase.** PostgreSQL is the system of record. The snapshot
  reader is retained for audit only.
