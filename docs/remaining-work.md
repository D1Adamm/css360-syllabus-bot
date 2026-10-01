# Remaining work

What is genuinely unfinished. Everything not listed here is implemented and
covered by tests — if you are looking for a feature and it is not below, look for
it in the code before building it.

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
  exists but is launched and registered by hand rather than through the queue.
Publication no longer needs Tillicum: an administrator activates a version
from Admin → Models once the VM serves it. `promote_qlora_adapter.sh` still
records publications too, as the legacy path.

Still by hand, deliberately, wherever training runs: deciding that a trained
version should be served (install and map it on the VM, then activate it).

---

## 7. Serving and concurrency (findings 2026-09-30, redesign deferred)

A production test ("When is the class") returned a RAG answer after a long
wait while Base, Fine-Tuned and Fine-Tuned + RAG showed "temporarily
unavailable". The redesign was deliberately deferred until after the classroom
deployment; these are the findings from the code, not yet confirmed by logs.

- **All four conditions share one Ollama on one 8-core CPU, behind two locks
  that do not know about each other.** The backend lock
  (`ollama_coordination.py`) serialises Base, RAG and automatic starter
  generation (qwen3:8b, up to 3072 tokens, 300 s per call). The fine-tuned
  service (`training/inference_service/ollama_service.py`) has its own lock and
  calls the same Ollama. Embeddings take no lock.
- **The browser schedules each student's four requests** (Base then RAG in
  sequence, both fine-tuned requests alongside), and nothing coordinates across
  students.
- **Timeouts are inconsistent.** Base/RAG allow 120 s from *after* the backend
  lock is acquired, so waiting for the lock is unbounded (only Nginx ends it).
  Fine-Tuned/Fine-Tuned + RAG allow 120 s *including* the wait behind the
  service's lock. Embeddings allow 60 s. Every failure reaches the student as
  the same "temporarily unavailable".
- **A disconnect cancels nothing**: the handlers return plain JSON, so an
  abandoned request keeps its lock and its CPU.
- **Latency was only ever measured for the fine-tuned conditions in
  isolation**; the mix the Compare page produces has not been measured.

Leading hypotheses, strongest first: a starter-generation job holding the lock
and the CPU (it starts automatically after a course's first syllabus upload);
cross-model CPU contention on 8 cores; model eviction/reload under memory
pressure; a fine-tuned configuration fault. To confirm, compare the Ollama
journal (`sudo journalctl -u ollama --since … --until …`) with the backend and
fine-tuned service status codes for the moment of the failure.

The proposed follow-up — one priority scheduler across all generation,
separate queue-wait and generation timeouts, backpressure, `keep_alive` on the
backend's calls, a backend comparison job so refresh recovery works, and
per-condition timing logs without question text — is described in the
2026-09-30 plan; measure N=1 versus N=2 concurrency with a classroom load probe
before choosing. Also open: how often `classroom-concise-v1` answers reach the
128-token cap (`docs/css360-model-evolution.md` §13).

## Known issues

- **Compare can answer "temporarily unavailable" under load** — §7. Seen in
  production (findings dated 2026-09-30); the root cause is not yet confirmed
  and the redesign is deferred.
- **A course with no activated version still follows `current_version`.**
  Kept deliberately for the rollout of activation (see
  [architecture.md](architecture.md#registered-published-served)). For such a
  course, a training run that registers `v2` moves fine-tuned requests to `v2`
  at once, and the VM refuses a version it does not map. The rollout:
  1. deploy activation (Admin → Models → **Activate**);
  2. activate the version each served course already uses — CSS 360D `v1`
     and CSS 360E `v1`, and any other course Admin → Models shows as
     "No activated version" — and re-run `verify_finetuned_production.py`;
  3. only then remove the fallback, so inference requires an activated
     version and registration alone can never move traffic.
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
