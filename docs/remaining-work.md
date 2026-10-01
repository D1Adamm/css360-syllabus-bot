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
from Admin → Models once the VM serves it, and only an activated version
answers — finishing or registering training never moves traffic on its own.
`promote_qlora_adapter.sh` still records publications too, as the legacy
path.

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

- **Measure the VM** with `scripts/classroom_load_test.py` after deploying,
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

## Known issues

- **Compare under classroom load** — §7. Root cause found and fixed in code
  2026-10-01; the second Ollama and the VM load test are deployment steps
  ([classroom-capacity.md](classroom-capacity.md)).
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
