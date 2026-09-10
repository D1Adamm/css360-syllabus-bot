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
- **No structured application logging or alerting.** A failed starter-seed job or
  an expired serving session is discoverable by looking, not by being told.
- **Course deletion does not touch the filesystem artifacts or the cluster.**

`training/cleanup_training_outputs.sh` covers cluster disk only, is dry-run by
default, and never proposes a published adapter.

## 5. Backend scalability

Syllabus artifacts and embedding indexes are written to the VM's local disk. One
backend process owns them, so the backend cannot be run as more than one
instance without moving that storage. Not a problem at classroom scale; it is the
first thing to hit if this grows.

## 6. Fine-tuned inference without Tillicum

Fine-tuned inference used to require a GPU session opened by hand, because the
tunnel authenticates to UW and two-factor is deliberately not automated. That
made the Fine-Tuned paths unavailable outside a scheduled session.

The CSS 360 v2 adapter now runs on the UWB VM's CPU through Ollama, and
`training/inference_service/ollama_service.py` serves it behind the same
`/health` and `/generate` contract the GPU service answers, on the same
`127.0.0.1:9001`. The backend client is unchanged. Courses are mapped to Ollama
models by configuration, so CSS 350 is one more entry once its adapter is
converted.

Training can now run on the VM as well: `train_qlora.py --cpu` and
`training/start_cpu_qlora_training.sh` run the same recipe (same split, chat
formatting, LoRA and NF4 configuration) with float32 compute on the VM's
cores, writing the cluster's output layout, and neither device mode ever
falls back to the other. See "CPU training on the UWB VM" in
`training/README.md`.

Still open: converting adapters to GGUF is a by-hand step; latency on the VM
under classroom load has not been measured; the training queue worker still
submits to Tillicum, so a CPU run is launched and registered by hand; and the
Tillicum GPU service remains the known-good baseline and fallback, so nothing
has been removed from it. See the "UWB VM" section of
`training/inference_service/README.md`.

---

## Known issues

None recorded. (The professor overview's hardcoded model status, listed here
earlier, was fixed: the page reads the registry and request records now.)

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
