# Remaining work

What is genuinely unfinished. Everything not listed here is implemented and
covered by tests — if you are looking for a feature and it is not below, look for
it in the code before building it.

Ordered by what blocks real classroom use.

---

## 1. Authentication and access control

**The largest gap, and the one everything else waits on.**

There is no sign-in. The role switcher in the header is a development control: it
changes navigation and vocabulary, grants nothing, and every route is reachable
by URL in any role. Anyone with a link can open any course as an admin.

What this blocks:

- A professor cannot be shown only their own courses.
- Nothing can be attributed to a person, so contributions and evaluations are
  anonymous whether or not that is wanted.
- Admin surfaces expose cluster detail — artifact references, Slurm job ids,
  compute hostnames — to anyone who visits `/admin`.

The backend is ready for it in one respect: the training-queue router already
authenticates its caller with a shared worker token, kept deliberately separate
from any future browser session so that adding user auth does not disturb the
cluster's credential.

## 2. Enrolment, join codes, and rosters

`/professor/course/:courseId/invite` renders an explanation and writes nothing.
There is no roster table, no join-code issue or redemption, and no way to scope
a student to a course.

Depends on authentication.

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
- **No course deletion path in the UI.** The schema cascades correctly from
  `courses`, but nothing exposes it, and cascading does not touch the filesystem
  artifacts or the cluster.

`training/cleanup_training_outputs.sh` covers cluster disk only, is dry-run by
default, and never proposes a published adapter.

## 5. Backend scalability

Syllabus artifacts and embedding indexes are written to the VM's local disk. One
backend process owns them, so the backend cannot be run as more than one
instance without moving that storage. Not a problem at classroom scale; it is the
first thing to hit if this grows.

## 6. Optional experiment: fine-tuned inference without Tillicum

Fine-tuned inference requires a GPU session opened by hand, because the tunnel
authenticates to UW and two-factor is deliberately not automated. That makes the
Fine-Tuned paths unavailable outside a scheduled session.

An experiment worth running later: serve a merged or quantised adapter on the
UWB VM's CPU and measure whether latency is tolerable for classroom use. If it
is, the fine-tuned paths stop depending on a person being present.

**Explicitly not started, and not a criticism of the current design.** Tillicum
remains the known-good baseline for both training and inference, and nothing
should be removed from it on the strength of an untested alternative.

---

## Known issues

**`CourseOverviewPage` hardcodes "Course model — Not available yet".**
`src/pages/professor/CourseOverviewPage.tsx` renders that string as a constant,
while `ProfessorModelPage` reads the real registry and request state on the same
course. A professor whose model is ready sees the correct state on one page and a
stale one on the other. A display bug, not a data bug — nothing is written
incorrectly.

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
