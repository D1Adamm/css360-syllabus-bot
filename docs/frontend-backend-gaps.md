# Frontend features waiting on backend support

The redesigned professor experience is complete except for three areas that
have no backend to talk to. Rather than fake persistence, each one ships as a
polished integration boundary: the layout is settled, the copy is honest, and
nothing is written anywhere.

This document is the handoff. Each section states what the UI already renders,
what it needs, and the exact request/response shape it expects.

---

## 1. Student enrolment and course join codes

**Where:** `/professor/course/:courseId/invite` (`InviteStudentsPage`)

**Today:** shows the real course URL with a copy button, plus an inert preview
of where a join code and QR code will sit. States plainly that access is not
restricted.

**Blocked by:** there are no user accounts, no authentication, and no course
membership records anywhere in the system.

**Needed:**

```
POST   /api/courses/{courseId}/join-codes        -> { code, expiresAt }
GET    /api/courses/{courseId}/join-codes        -> { code, expiresAt, uses } | 404
DELETE /api/courses/{courseId}/join-codes/{code} -> 204
POST   /api/join                                 { code } -> { courseId }
GET    /api/courses/{courseId}/members           -> { count }
```

A join code only means something once a request can be attributed to a person,
so this depends on authentication landing first.

**Deliberately not done:** no code is generated client-side and stored in
`localStorage`. A code that grants nothing, but looks like it does, is worse
than no code.

---

## 2. Course model requests

**Where:** `/professor/course/:courseId/model` (`ProfessorModelPage`),
`/professor/models` (`ModelsHubPage`)

**Today:** reports the approved-example count, which *is* verifiable, and says
requesting a model is not available yet. No request button submits anything.

**Blocked by:** there is no record of a fine-tuning request and no training job
state. Training currently runs through Slurm scripts on Tillicum, outside this
application entirely.

**Needed:**

```
POST /api/courses/{courseId}/model-requests   { note? } -> { requestId, state, requestedAt }
GET  /api/courses/{courseId}/model-requests   -> { requests: [...] }
GET  /api/courses/{courseId}/model            -> { state, updatedAt, modelVersion? }
```

`state` should be one of `requested | preparing | ready | failed`, matching
`CourseModelState` in `src/lib/modelStatus.ts`.

**Important — why `/fine-tuned/health` is not used here:** that endpoint
describes a single shared inference service. It reports whether *some* adapter
is loaded, not which course it belongs to. Reading it as "this course's model
is ready" would be correct for at most one course and wrong for every other, so
`getCourseModelStatus()` returns `unknown` and reads nothing.

---

## 3. Per-course model registry

**Where:** `/admin/models` (`AdminModelsPage`)

**Today:** shows the live fine-tuned service (status, model, adapter loaded,
host) and an explicit empty state for version history.

**Blocked by:** adapter versions, promotion, and rollback are handled by
`training/promote_qlora_adapter.sh` with no record the application can read.

**Needed:**

```
GET  /api/models                     -> { versions: [{ id, courseId, createdAt, active }] }
POST /api/models/{versionId}/promote -> { active: true }
POST /api/models/{versionId}/rollback
```

---

## Notes on things that are *not* gaps

- **Approved / pending example counts are real.** They come from
  `GET /api/courses/{courseId}/seeds` and are computed in
  `src/lib/exampleCounts.ts`. They are deliberately not derived from
  `GET /api/courses/{courseId}/seeds/approved-export-status`: an export is a
  training artefact that can be stale, missing, or newer than the current review
  state, and "48 approved" must mean the review state right now.
- **Export and train/validation split are real** and live in Admin only
  (`/admin/training`). They were moved off the professor review page, where they
  exposed dataset mechanics and absolute server paths.
- **Course creation, syllabus upload, and the failure rollback are real** and
  unchanged.
