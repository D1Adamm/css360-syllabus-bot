# Deployment

How a change reaches the UWB VM and the Tillicum checkout.

Everything here is run by hand. There is no CI, no automated deploy, and nothing
in this repository triggers a deployment.

Values below reflect the deployed setup as it is represented in this repository —
the backend's systemd unit name and port come from `scripts/aiswe_finetuned.sh`
and `scripts/start_finetuned_tunnel.sh`, the fine-tuned unit from
`training/inference_service/aiswe-finetuned.service`, the Nginx document root
and the SELinux relabel from the publish step that has been used against the
VM. Anything not represented anywhere in the repository is not documented here.

---

## The deployed shape

| Piece | Where |
| --- | --- |
| Static frontend | Nginx, document root `/usr/share/nginx/html/` |
| Backend | uvicorn on `127.0.0.1:8001`, systemd **user** unit `aiswe-backend` |
| Reverse proxy | Nginx forwards `location /api/` to the backend |
| Database | PostgreSQL on the VM, DSN in `backend/.env` |
| Generation | Ollama on the VM |
| Fine-tuned inference | `training/inference_service/ollama_service.py` on `127.0.0.1:9001`, systemd **user** unit `aiswe-finetuned`, against a second Ollama on `127.0.0.1:11435` (system unit `ollama-finetuned`; see [classroom-capacity.md](classroom-capacity.md)) |
| Repository | `~/css360-syllabus-bot` on the VM, `/gpfs/projects/simswe/$USER/css360-syllabus-bot` on Tillicum |
| Reference copies | [templates/](templates/README.md): the backend unit, the Nginx site and the `backend/.env` shape as verified on 2026-10-31 |

**Nginx forwards only `/api/`.** This is why `VITE_API_BASE_URL` carries the
`/api` prefix and the frontend clients write paths below it, and why the backend
serves a handful of root-level aliases (`/health`, `/rag/generate`, …) that are
reachable only on the VM itself, not through the proxy.

---

## Branching

`main` is the deployed branch. Review on a feature branch, merge, then pull
`main` on both hosts — never deploy a branch.

```bash
git checkout -b <branch> && git add -A && git commit -m "<message>"
```
```bash
git push -u origin <branch>
```

After review:

```bash
git checkout main && git merge --no-ff <branch> && git push origin main
```

If the SSH remote is refused from the shell you are in (`Permission denied
(publickey)`), push the same fast-forward over HTTPS with the `gh` credential
helper, never with `--force`:

```bash
GIT_CONFIG_GLOBAL=/dev/null git -c credential.helper='!gh auth git-credential' push https://github.com/D1Adamm/css360-syllabus-bot.git main:main
```

Read the live remote head first the same way (`ls-remote … refs/heads/main`)
and confirm `origin/main` is an ancestor of your `main`.

---

## UWB VM

The deployment of `568a480` on 2026-10-31 followed this order and is recorded
in [verification-history.md](verification-history.md#deployment-of-568a480-2026-10-31).
Everything up to the backend restart leaves production serving the old code,
so any step before it is a safe place to stop.

### Before pulling

Confirm the VM is idle and in VM mode (expect the current commit, only
untracked local files in `git status`, three `active`, `MODE: VM (normal)`,
and no generation in the last quarter hour):

```bash
cd ~/css360-syllabus-bot && git rev-parse --short HEAD && git status --short
```
```bash
systemctl --user is-active aiswe-backend aiswe-finetuned aiswe-gpu-failback.timer
```
```bash
./scripts/classroom_gpu_mode.sh status | head -4
```
```bash
journalctl --user-unit aiswe-backend --since '-15 min' --no-pager -o cat | grep -c 'generation condition='
```

Back up what a rollback or a rebuilt host would need. The directory is private
to the account; keep the same shell so `$B` stays set:

```bash
B=~/backups/deploy-$(date -u +%Y%m%dT%H%M%SZ) && mkdir -p "$B" && chmod 700 "$B" && echo "$B"
```
```bash
cp -a $(git ls-files --others --exclude-standard) "$B/" 2>/dev/null; cp -p backend/.env "$B/backend.env" && chmod 600 "$B/backend.env" && cp -p ~/.config/aiswe/finetuned.env "$B/finetuned.env" && cp -p ~/.config/aiswe/generation-mode.json "$B/" && cp -a ~/.config/systemd/user/aiswe-*.service ~/.config/systemd/user/aiswe-*.timer "$B/" && cp -a ~/.local/state/css360-syllabus-bot "$B/state"
```
```bash
sudo cat /etc/nginx/conf.d/aiswe.conf > "$B/aiswe.conf" && sudo systemctl cat ollama ollama-finetuned > "$B/ollama-units.txt" && git rev-parse HEAD > "$B/previous-commit.txt"
```
```bash
(set -a && source backend/.env && set +a && pg_dump "$DATABASE_URL" -Fc -f "$B/db-before.dump") && ls -la "$B"
```

The first line copies every untracked file in the checkout (local experiments
such as `training/train_cpu_qlora_experiment.py`). Model artifacts (the Ollama
store, `~/model_artifacts`, `data/`, `local-artifact-backup/`) are not part of
a deployment and are left alone by every step here.

Review what will arrive, and whether it needs a frontend rebuild (`src/`,
`package.json`), a migration (`backend/db/`), a dependency install
(`backend/requirements.txt`) or a fine-tuned-service restart
(`training/inference_service/`):

```bash
git fetch origin && git log --oneline HEAD..origin/main && git diff --name-only HEAD origin/main
```

### Pull

```bash
git pull --ff-only origin main && git rev-parse --short HEAD && git status --short
```

A fast-forward pull changes tracked files only. It never touches untracked
files, ignored files (`backend/.env`, `.env.local`) or any model artifact.
Never run `git clean`, `git stash -u` or `git checkout .` on the VM. Afterwards
`git status` must show the same untracked files as before.

### Database migrations, if any

Migrations in `backend/db/migrations/` upgrade a database that already has data.
A fresh database gets everything from `backend/db/schema.sql` instead and needs
no migration.

Load the DSN from the file the backend already uses, rather than retyping it:

```bash
set -a
source backend/.env
set +a
```
```bash
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f backend/db/migrations/<migration>.sql
```

Every migration is written to be idempotent (`ADD COLUMN IF NOT EXISTS`,
`CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`), so re-running one is
a no-op rather than an error. `tests/test_schema_files.py` asserts both
properties for every migration file.

| Migration | Adds |
| --- | --- |
| `001_training_provenance_and_serving.sql` | Training provenance columns and `serving_sessions` |
| `002_auth_identity.sql` | `users`, `course_memberships`, `invitations`, `invitation_course_grants`, `participants`, `auth_sessions`, `admin_actions`; nullable `participant_id` on `evaluations` and `seed_examples`; `created_by` on `courses`. No existing row changes value |
| `003_evaluation_provenance_and_dataset_pinning.sql` | Nullable `responses` JSONB on `evaluations` (the four answers a rating was made against, with model, version and engine), an index on `evaluations (course_id, participant_id, run_id)` for the one-rating-per-run check, and nullable `dataset_sha256` on `training_runs` (the prepared files a queued run is pinned to). No existing row changes value; older rows simply carry NULL |
| `004_classroom_answer_cache.sql` | Three new tables for classroom preparation ([classroom-preparation.md](classroom-preparation.md)): `classroom_questions` (which course questions staff selected), `answer_cache` (one stored answer per course, approach, question and configuration) and `classroom_preparation` (how a course's last preparation run ended). No existing table or row changes. On branch `feat/compare-answer-cache`; not applied anywhere yet |

### Frontend

```bash
npm ci
```
```bash
npm run build
```
```bash
sudo cp -a dist/. /usr/share/nginx/html/
```
```bash
sudo restorecon -R /usr/share/nginx/html
```
```bash
sudo nginx -t
```
```bash
sudo systemctl reload nginx
```

`restorecon` restores SELinux labels on the copied files; without it Nginx may be
unable to read what was just deployed. `nginx -t` before `reload` so a bad
configuration is caught while the old one is still serving.

### Backend

Run both suites on the VM first (next section); they prove the pulled code
under the VM's own Python and libraries before anything restarts. Then, with no
class in progress, restart only the backend. An in-flight generation at that
moment fails once; a retry works.

```bash
systemctl --user restart aiswe-backend && sleep 3 && curl -s http://127.0.0.1:8001/api/health && echo && systemctl --user show aiswe-backend -p ActiveState -p NRestarts
```
```bash
journalctl --user-unit aiswe-backend --since '-2 min' --no-pager -o cat | grep -iE 'traceback|error|Application startup complete' | tail -5
```

Expect `{"status":"ok","service":"syllabus-model-lab-backend"}`, `active`,
`NRestarts=0`, `Application startup complete` and no traceback. Nothing else
restarts for a code deployment: not `ollama`, not `ollama-finetuned`, not
Nginx, and no `daemon-reload` unless a unit file changed. Then run the checks
under [After a deployment](#after-a-deployment).

A user unit, not a system one — so it is managed without root, and it needs
lingering enabled for the account if it is to survive logout.

When a pull changes `training/inference_service/`, restart the VM-local
fine-tuned service too — it runs from the checkout and only reads its code at
start:

```bash
./scripts/aiswe_finetuned.sh restart
```

### First deployment of authentication

The first deployment that carries migration 002 turns on sign-in for every
browser route. There is no bypass flag; the order below is the whole cutover,
and every step before the bootstrap link is safe to do ahead of time.

1. `git pull origin main`, then apply `backend/db/migrations/002_auth_identity.sql`
   as above. The running backend ignores the new tables until it is restarted.
2. Add `APP_PUBLIC_ORIGIN=https://aiswe.uwb.edu` to `backend/.env`. Nothing else
   in the authentication block needs a value on the VM.
3. Build and publish the frontend, restart `aiswe-backend`, check `/api/health`.
   From this moment every route except health, sign-in and join answers 401 to
   a browser without a session.
4. Mint the first administrator invitation and open it within the hour:

   ```bash
   cd ~/css360-syllabus-bot/backend && .venv/bin/python scripts/bootstrap_admin_invite.py --origin https://aiswe.uwb.edu
   ```

   Choose an email address, a display name and a password on the page it prints.
   The script refuses once an administrator exists; `--force` is for lockout
   recovery only, and both are recorded in `admin_actions`.
5. In **Admin → People**, create an instructor invitation per professor with
   their courses ticked, and send each link yourself. Links are shown once.
6. On each course's **Invite students** page, create the class code and put
   the join page and the code on the board (or the direct link in Canvas).
7. Verify from the VM:

   ```bash
   curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8001/api/db/courses
   ```

   Expect `401`. The training worker is unaffected: its header token reaches
   `/api/training-queue` exactly as before, and a dry run of the queue
   (`./training/run_training_queue.sh --once --dry-run` on Tillicum) proves it.

### Nginx notes for sessions

The Nginx configuration is not in this repository; these are the properties it
must have, checked with `sudo nginx -T`. The live `/api/` block was read on
2026-10-31 and had every one of them; it is recorded in
[verification-history.md](verification-history.md#production-vm-configuration-2026-10-31).

- `try_files $uri /index.html` (or equivalent) must already be in place for
  the SPA; `/join`, `/join/<code>`, `/invite/<token>` and `/login` are
  frontend routes served by `index.html`.
- **Forward the client address, or the whole class shares one throttle.**
  Join-code and login failures are limited per client. Uvicorn (0.49 or later, 0.52.0 on the VM as of 2026-10-31; its
  defaults `--proxy-headers` on and `--forwarded-allow-ips 127.0.0.1`) trusts
  `X-Forwarded-For` only from the loopback peer, i.e. from this Nginx, and takes
  the entry Nginx appended — a value a browser adds itself is ignored. Nginx
  does not send the header unless told to. The `location /api/` block needs:

  ```nginx
  location /api/ {
      proxy_pass http://127.0.0.1:8001;
      proxy_http_version 1.1;
      proxy_set_header Host $host;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      proxy_set_header X-Forwarded-Proto $scheme;
      client_max_body_size 11m;
      proxy_read_timeout 300;
      proxy_connect_timeout 60;
  }
  ```

  Without the `X-Forwarded-For` line every browser reaches the backend as
  `127.0.0.1`, and ten mistyped codes anywhere in the room lock the join page
  for everyone for ten minutes. If a load balancer ever sits in front of the
  VM, add `set_real_ip_from <its address>; real_ip_header X-Forwarded-For;` so
  `$remote_addr` is the student's address rather than the balancer's.
- **Give the generation queue its 300 s.** A generation request may wait up to
  150 s in the backend's queue (`GENERATION_QUEUE_TIMEOUT_SECONDS`) and then
  generate for up to 120 s, so the backend answers within 270 s and tells the
  student "busy" itself. Nginx's default `proxy_read_timeout` is 60 s, which
  would turn every longer wait into a bare 504 while the backend kept
  generating. `proxy_read_timeout 300;` in the `/api/` location is what the
  queue is sized for (`backend/app/generation_queue.py`); if the queue timeout
  is ever raised, raise this with it, keeping queue timeout + 120 s below it.
- **Let a 10 MB syllabus through.** The backend accepts syllabus files up to
  10 MB (`MAX_SYLLABUS_BYTES` in `backend/app/syllabus_upload.py`) and says so
  when one is larger. Nginx refuses any request body over its
  `client_max_body_size` first — **1 MB when it is not set** — with a bare 413
  the application never sees. `11m` in the `/api/` location leaves room for the
  multipart envelope around a 10 MB file, so the backend's limit is the one
  professors meet. The directive may be set in the `http`, `server` or
  `location` context, and the innermost one that applies wins, so check every
  occurrence rather than the first:

  ```bash
  sudo nginx -T 2>/dev/null | grep -nE '^# configuration file|^\s*(server_name|location|client_max_body_size|proxy_read_timeout)\b'
  ```

  That lists each file, each server and location, and every limit, in order;
  a `/api/` location with no `client_max_body_size` above or inside it is on
  the 1 MB default. `proxy_read_timeout` is in the same listing because an
  upload is processed before it answers (extraction and embedding, typically
  well under a minute): if Nginx gives up first (60 s by default; the `/api/` block above sets 300 s), the browser
  shows an error while the backend finishes and records the syllabus, and a
  reload shows it ready.
- Nothing rewrites cookies. The backend sets `Secure; HttpOnly; SameSite=Lax;
  Path=/api` itself, from configuration rather than from proxy headers.

### Tests on the VM

Safe to run. The suite cannot reach the production database: under pytest,
`backend/.env` is not read and `DATABASE_URL` is ignored outright.

```bash
backend/.venv/bin/python -m pytest backend/tests -q
```
```bash
backend/.venv/bin/python -m pytest training scripts -q
```

The second suite includes the fallback-script tests, which run the real
operator scripts against fake `systemctl`, `ssh` and `curl` placed first on
`PATH`; they touch no real unit. Both suites must pass before the backend is
restarted.

That barrier exists because it was once not true — see
[tillicum-operations.md](tillicum-operations.md#test-isolation) and
`backend/tests/test_test_isolation.py`.

---


## Fine-tuned inference on the VM

Fine-Tuned and Fine-Tuned + RAG are answered on the VM by
`training/inference_service/ollama_service.py`, run as the systemd **user**
unit `aiswe-finetuned` on `127.0.0.1:9001`, asking the VM's own Ollama for one
model per course and version. Nothing outside the VM is involved: no Tillicum
allocation, no SSH tunnel, no Duo prompt, no laptop, no `caffeinate`, and no
terminal that has to stay open. The unit starts at boot and restarts on
failure; lingering must be on for the account, or user units stop at logout.

Everything is driven by one helper. Each command is idempotent and none of
them kills a process it did not start.

```bash
./scripts/aiswe_finetuned.sh install        # render + install the unit, create ~/.config/aiswe/finetuned.env, set FINETUNED_SERVICE_URL
```
```bash
./scripts/aiswe_finetuned.sh check          # read-only: unit, mapping, port owner, tunnel, Ollama, health
```
```bash
./scripts/aiswe_finetuned.sh start          # also: stop, restart, status, logs
```

`install` renders `training/inference_service/aiswe-finetuned.service` with
this checkout's path, writes it to `~/.config/systemd/user/`, creates the
environment file from `scripts/finetuned.env.example` if it does not exist
(mode 600, empty mapping, `OLLAMA_BASE_URL=http://127.0.0.1:11435` for the
second Ollama), sets `FINETUNED_SERVICE_URL=http://127.0.0.1:9001`
in `backend/.env` if it is not already that, and enables the unit. It does
not start it. Lingering, once per account:

```bash
loginctl enable-linger $USER
```

### Installing a completed adapter

The registry in PostgreSQL decides which version a course serves. The VM has
to hold an Ollama model for exactly that version, built from the adapter the
version's `artifact_ref` names. One command validates the adapter, converts it
to GGUF, builds the versioned Ollama model, verifies the tag exists, and prints
the mapping entry. It never registers, publishes or promotes anything.

The base model must come from the VM's local Hugging Face snapshot. With only
`--base-model-id` (the default) the installer hands that id to llama.cpp's
converter, which then tries to reach Hugging Face — even when the model is
fully cached — and the conversion fails on the VM. Pass the snapshot directory
with `--base-model-path` instead; the installer does not search the cache for
you, because picking a snapshot (several revisions, a partial download, a
non-default `HF_HOME`) is a decision an operator should see:

```bash
BASE_SNAPSHOT=~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/$(cat ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/refs/main)
```
```bash
ls "$BASE_SNAPSHOT"/config.json
```
```bash
backend/.venv/bin/python scripts/install_finetuned_adapter.py --course <courseId> --version <vN> --tag <tag> --adapter <adapter-dir> --base-model-path "$BASE_SNAPSHOT" --smoke --dry-run
```

**`--tag`** is the Ollama model name, and Ollama names are shared by every
course. For a legacy course id (`css-360-winter-2026-a7rp`) it defaults to
`css360-ft-<vN>`. A newer id (`css360d-fall-2026-q0ne`) has no derivable short
name and must pass one — production uses `css360d-v1` and `css360e-v1`. Choose
a tag unique to the course and version: `--replace` rebuilds whatever model
already has that name, whichever course it serves.

Drop `--dry-run` to run it. Conversion uses llama.cpp's
`convert_lora_to_gguf.py` from the checkout at `~/model_artifacts/llama.cpp`
(or `--llama-cpp-dir`) under that checkout's own `.venv/bin/python`, which holds
`gguf`, `torch`, `transformers` and `safetensors`. The installer runs that
interpreter by its venv path rather than the system Python it links to, so the
venv's packages are used; `--converter-python` overrides it. The dry run prints
the conversion command, and its first word should be that `.venv/bin/python`.
If the VM cannot convert, convert on Tillicum and pass the result with
`--gguf <file>`; the GGUF and the Ollama model always end up on the VM. An
existing GGUF or tag for the same course and version is refused without
`--replace`. Artifacts land under `~/model_artifacts/<courseId>/<vN>/` with an
`install-record.json` (hashes, converter flags, Ollama digest).

### Updating the mapping

```bash
./scripts/aiswe_finetuned.sh set-mapping <courseId> <vN> <ollamaTag>
```
```bash
./scripts/aiswe_finetuned.sh restart
```

`set-mapping` edits `~/.config/aiswe/finetuned.env`, validates the whole
mapping with the service's own parser, refuses to re-point a version that is
already mapped to a different model unless `--replace` is given, and warns if
Ollama does not have the tag yet. `--dry-run` prints the result without
writing. The unit reads the file only at start, so `restart` applies it; the
unit's preflight refuses to start on a malformed mapping.

### Verifying

```bash
AISWE_VERIFY_PASSWORD='<admin password>' backend/.venv/bin/python scripts/verify_finetuned_production.py --admin-email <admin email> --course css360d-fall-2026-q0ne --course css360e-autumn-2026-c08m
```

Always name the courses in class. Without `--course` the script checks the
legacy CSS 350 and CSS 360 ids (`KNOWN_COURSES` in
`scripts/lib/finetuned_production_checks.py`), once per course:

```bash
AISWE_VERIFY_PASSWORD='<admin password>' backend/.venv/bin/python scripts/verify_finetuned_production.py --admin-email <admin email> --course <courseId> --course <otherCourseId>
```

One PASS/FAIL/SKIP line per check, exit 1 on any FAIL: the unit is enabled and
active, a python process (not `ssh`) owns port 9001, Ollama has every mapped
model, `/health` reports `status=ok` and `adapterLoaded=true` with every
mapped course and version, direct `/generate` answers for each mapped course
with the right course and version, and, through the backend as an
administrator, Fine-Tuned, Fine-Tuned + RAG, Base and RAG for each course
checked.
Without credentials the backend checks are SKIP, never PASS; the password is
read from the environment (or a prompt), never stored, and the session is
logged out. No answer text is printed. This is the check to run after a
deploy and after a reboot.

A small latency probe, for the classroom question rather than for load:

```bash
AISWE_VERIFY_PASSWORD='<admin password>' backend/.venv/bin/python scripts/finetuned_latency_probe.py --admin-email <admin email> --out latency.json
```

For load (many students at once, all four conditions), use
`scripts/classroom_load_test.py` instead; see
[classroom-capacity.md](classroom-capacity.md), which also covers the second
Ollama server the course models are served from.

Defaults: both courses, both fine-tuned modes, one first request per cell,
then two requests each at concurrency 1 and 2. Concurrency 4 needs
`--allow-concurrency-4` and is the ceiling. The VM's Ollama is one shared CPU
process and the service serialises generations, so concurrency measures
queueing; the second student waits for the first.

### After a deployment

Read-only unless marked. The deployment of `568a480` passed every one of these.

1. `./scripts/aiswe_finetuned.sh check`: every line PASS, mapping unchanged.
2. `./scripts/classroom_gpu_mode.sh status`: `MODE: VM (normal)`, `backend ok`,
   `fine-tuned ok (… -> http://127.0.0.1:11435)`, watchdog `timer RUNNING`,
   no WARNING.
3. `./scripts/status_finetuned_tunnel.sh`: `Overall: Fine-Tuned / Fine-Tuned +
   RAG available`, exit 0.
4. The verify script above for the taught courses: every line PASS. It sends
   each course's **activated** version through the backend, so all four
   conditions are exercised for every course named.
5. In a browser signed in as an administrator: `/api/admin/generation-mode`
   answers `"mode":"vm"` with the fine-tuned routes on `http://127.0.0.1:9001`;
   `/api/admin/generation-queue` answers `maxConcurrency 1`, `maxWaiting 160`,
   `queueTimeoutSeconds 150`.
6. Admin → Models shows the activated version of each taught course. Pressing
   **Activate** on the version that is already active answers `unchanged` and
   moves nothing; the servability probe ran.
7. Admin → student preview: one Compare for each taught course, all four
   answers. Then confirm each course answered with its own model:
   ```bash
   journalctl --user-unit aiswe-finetuned --since '-15 min' --no-pager -o cat | grep -oE 'css360[de]-v1:latest' | sort | uniq -c
   ```
8. Switch Admin ↔ Professor view; the My courses filter; the Ask questions tab.
   In preview, submit one rating: it is accepted with a `preview-…` id and
   Admin → Research counts do not change.
9. Worker callbacks (writes nothing): pick a run in state `succeeded` from
   Admin → Training jobs and post a submission report for it with the worker
   token read from the env file, never echoed. Expect `HTTP 409`, "already
   succeeded … Nothing was recorded" (or the "superseded" refusal if an admin
   retry has moved the request on), and the run unchanged.
   ```bash
   set -a && source backend/.env && set +a && curl -s -o /dev/stderr -w '\nHTTP %{http_code}\n' -X POST -H "X-Training-Worker-Token: $TRAINING_WORKER_TOKEN" -H 'Content-Type: application/json' -d '{"jobId":"0"}' http://127.0.0.1:8001/api/training-queue/courses/<course id>/runs/<succeeded run id>/submitted
   ```
10. `./scripts/gpu_failback_watchdog.sh status`: timer running, last check
    seconds old.

Not part of an ordinary deployment: rehearsing the emergency fallback or a
classroom GPU mode drill. Both need a Tillicum allocation and a Duo prompt and
briefly move where the fine-tuned conditions are served; schedule them.

### Making a new version the one students get

Installing and mapping a version makes it *servable*; the backend still sends
the course's *active* (published) version. Install and map first, restart,
then in **Admin → Models** press **Activate** on that version and confirm.
Activation asks the fine-tuned service's `/health` first and refuses (nothing
written) unless it lists that exact course and version, so the wrong order
fails safely. Rollback is the same button on the earlier version, which must
still be mapped. Each activation is recorded in the audit trail.

A course with no activated version has no fine-tuned answers: there is no
fallback to the newest registered version. A course's first trained version
therefore serves nothing until it is installed, mapped and activated.

Tillicum's `training/promote_qlora_adapter.sh` still records a publication as
the legacy path ([tillicum-operations.md](tillicum-operations.md#per-course-serving)).
Its report is checked against the fine-tuned service exactly as activation is:
a version the VM does not map is refused, nothing is written, and the previously
published version keeps serving.

### Tillicum fallback, and back again

Classroom GPU mode (all four conditions on a Tillicum GPU for a class, with
an automatic failback watchdog) is separate from this fallback; see
[classroom-gpu-mode.md](classroom-gpu-mode.md#15-deploying-from-github-to-aiswe)
for its deployment steps and [classroom-gpu-runbook.md](classroom-gpu-runbook.md)
for operating it.

The Tillicum GPU service is kept for emergencies and claims the same local
port through an SSH tunnel, so the two never run together. Normal operation
needs neither. Switching is explicit in both directions:

```bash
./scripts/start_finetuned_tunnel.sh --from-backend --stop-local    # to the fallback (Duo prompt)
```
```bash
./scripts/stop_finetuned_tunnel.sh                                 # back to normal, verified
```

`--stop-local` stops `aiswe-finetuned` through `aiswe_finetuned.sh stop
--for-fallback`, which records the stop and the course/version pairs the
service could serve (`~/.local/state/css360-syllabus-bot/ft-fallback-local.env`).
If the tunnel start then fails (Duo cancelled, the node never healthy, the
backend restart failing), it closes the tunnel it opened and restores the unit
before exiting. `stop_finetuned_tunnel.sh` closes the tunnel and runs
`aiswe_finetuned.sh restore`: start the unit, then require every recorded pair
to be servable again, or every mapped pair when there is no record. The record
is removed only then, so a failed restore can be re-run (`aiswe_finetuned.sh
restore`). It also restores a unit that is enabled but stopped with no record
(the older manual `aiswe_finetuned.sh stop`); `--keep-local-stopped` opts out.

Neither script needs an administrator session. `start_finetuned_tunnel.sh`
ends by checking what it can without one: the backend answers `/api/health`,
`backend/.env` names the tunnel as `FINETUNED_SERVICE_URL`, and the tunnel
still answers `/health`. The backend's own `/api/fine-tuned/health` has
required an administrator session since the authorization work (2026-09-04);
until 2026-10-31 both scripts still probed it bare, so a correct fallback
start failed at its last step and undid itself, and `status` reported a
healthy service as unavailable. The authenticated end-to-end check is
Admin → Models, which calls that route.

The two never compete for the port: `start_finetuned_tunnel.sh` refuses while
`aiswe-finetuned` is active unless given `--stop-local`, `aiswe_finetuned.sh
start` (and the unit's own preflight) refuses while the tunnel owns it, and
`stop_finetuned_tunnel.sh` starts nothing while a tunnel it could not close
still holds it. `./scripts/aiswe_finetuned.sh status` and `classroom_gpu_mode.sh
status` warn when the unit is stopped and no tunnel holds the port. The unit
stays enabled through a fallback, so a reboot returns the VM to local serving.

Classroom GPU mode is not this fallback and never stops `aiswe-finetuned`: it
tunnels on `:9101`, and the VM service stays up as the failback target.

### After a reboot

Ollama is a system service and comes up on its own; `aiswe-backend` and
`aiswe-finetuned` are user units and come up with lingering. The unit's
preflight waits for Ollama rather than failing while it starts. Confirm with:

```bash
./scripts/aiswe_finetuned.sh check && AISWE_VERIFY_PASSWORD='<admin password>' backend/.venv/bin/python scripts/verify_finetuned_production.py --admin-email <admin email> --course css360d-fall-2026-q0ne --course css360e-autumn-2026-c08m
```

Name the taught courses: without `--course`, the script checks the legacy
CSS 350 and CSS 360 ids (`KNOWN_COURSES` in
`scripts/lib/finetuned_production_checks.py`), not the ones in class.

---

## Tillicum

```bash
cd /gpfs/projects/simswe/$USER/css360-syllabus-bot && git pull origin main
```
```bash
mkdir -p /gpfs/projects/simswe/$USER/training_outputs/serving
```
```bash
chmod 600 .env.local
```
```bash
./training/run_training_queue.sh --once --dry-run
```

`chmod 600` matters: `.env.local` holds `TRAINING_WORKER_TOKEN` on a shared
project filesystem, and training jobs read it from there. See the secrets section
of [tillicum-operations.md](tillicum-operations.md#secrets).

The dry run confirms the checkout can reach the backend and see the queue. It
claims nothing and writes nothing.

---

## Configuration on each host

Neither `.env` file is in the repository. Copy the example and fill it in.

**UWB VM** — `backend/.env`, from `backend/.env.example`:

| Variable | Required | For |
| --- | --- | --- |
| `DATABASE_URL` | yes | Everything |
| `TRAINING_WORKER_TOKEN` | for training | The queue API. Unset ⇒ that router refuses every request with 503 |
| `FINETUNED_SERVICE_URL` | for fine-tuned paths | `http://127.0.0.1:9001`: the local `aiswe-finetuned` unit (set by `scripts/aiswe_finetuned.sh install`). The Tillicum tunnel script sets the same value when that fallback is used |
| `CORS_ALLOWED_ORIGINS` | yes | The site origin. Also the CSRF origin allowlist |
| `APP_PUBLIC_ORIGIN` | recommended | The site origin, for the bootstrap script's printed link |
| `AUTH_COOKIE_SECURE` | no | Default true. Never set false on the VM |
| `AUTH_*` lifetimes | no | Session and invitation lifetimes; defaults documented in `backend/.env.example` |
| `AUTO_STARTER_SEED_GENERATION` | yes | Set `false` explicitly. The code default is `true` (`backend/app/starter_jobs.py`), and a starter seed job holds the single generation slot for minutes; `backend/.env.example` ships `false` |
| `GENERATION_MAX_CONCURRENCY`, `GENERATION_MAX_WAITING`, `GENERATION_QUEUE_TIMEOUT_SECONDS` | no | The generation queue. Defaults `1`, `160`, `150` (`backend/app/generation_queue.py`), sized for Nginx's `proxy_read_timeout 300` |
| `FINETUNED_READINESS_CHECK_SECONDS` | no | How often the backend re-checks that every course's activated model version is one the fine-tuned service serves, logging a journal WARNING when it is not and INFO when it recovers (`backend/app/finetuned_readiness.py`). Default `300`; `0` disables the background check. The same assessment is `GET /api/admin/model-readiness` (Admin → Overview) |
| `OLLAMA_*` | yes | Base and RAG generation |

**UWB VM** — `.env.local` for the frontend build: `VITE_API_BASE_URL` must be the
site origin **plus `/api`**.

**Tillicum** — `.env.local`, from `.env.example`: `TRAINING_API_BASE_URL` and
`TRAINING_WORKER_TOKEN`, matching the backend's value.

`backend/.env` holds `DATABASE_URL` and the worker token and must be mode 600
(`chmod 600 backend/.env`; it was 644 until 2026-10-31). The shape production
sets, with placeholders for the secrets, is
[templates/backend.env.production.example](templates/backend.env.production.example).
Two values are worth checking against it: `CORS_ALLOWED_ORIGINS` should be the
site origin (the 2026-10-31 inspection found the Vite development origins still
listed, which is harmless only because `APP_PUBLIC_ORIGIN` also feeds the CSRF
allow-list), and `AUTO_STARTER_SEED_GENERATION=false` must be present because
the code default is `true`.

---

## Monitoring and troubleshooting

There is no alerting ([remaining-work.md](remaining-work.md#4-privacy-retention-and-monitoring));
these are the places to look.

- **A class in progress:** `journalctl --user-unit aiswe-backend -f | grep ' generation '`
  prints one line per generation with its queue wait; `busy` outcomes are
  refusals. The administrator endpoint `/api/admin/generation-queue` shows the
  live queue. Details: [classroom-capacity.md](classroom-capacity.md).
- **Serving health:** `./scripts/aiswe_finetuned.sh status` (unit, port owner,
  mapping, `/health`), `./scripts/classroom_gpu_mode.sh status` (mode, tunnel,
  every condition's destination, both Ollama servers, the watchdog),
  `./scripts/status_finetuned_tunnel.sh` (the emergency fallback). The
  backend's own `/api/fine-tuned/health` needs an administrator session;
  Admin → Models shows it.
- **GPU mode and failback:** `./scripts/gpu_failback_watchdog.sh status`, and
  [classroom-gpu-mode.md §12–§13](classroom-gpu-mode.md#12-monitoring) for
  Tillicum job, node and VM resource checks and the symptom-by-symptom
  troubleshooting list.
- **Training and the worker:** the outage and retry table in
  [tillicum-operations.md](tillicum-operations.md#outage-and-retry-behaviour);
  the cluster's own record under `training/state/`.
- **Nginx:** `sudo tail -n 100 /var/log/nginx/error.log` for upstream
  timeouts (`upstream timed out`) and oversize bodies (`client intended to
  send too large body`); the current access log for 5xx responses.
- **After a reboot:** everything comes back on its own (system Ollama units,
  user units with lingering); confirm with the commands under
  [After a reboot](#after-a-reboot).

---

## Rolling back

Application code, on the VM. `git reset --hard` moves tracked files only: the
untracked local files stay, and the backup taken before the pull holds copies
of everything else.

```bash
cd ~/css360-syllabus-bot && git reset --hard "$(cat "$B/previous-commit.txt")" && git rev-parse --short HEAD && git status --short
```

Then, as needed: restore `backend/.env` from the backup only if it was edited
for the deployment (`cp -p "$B/backend.env" backend/.env && chmod 600 backend/.env`;
the mode-600 change can stay); rebuild and republish the frontend only if
`src/` changed between the two commits; restart `aiswe-backend` and check
`/api/health` and `./scripts/aiswe_finetuned.sh check`. On Tillicum,
`git reset --hard <previous-commit>` in the cluster checkout. Leave GitHub
`main` as it is and fix forward on a new branch rather than rewriting pushed
history.

**Model versions do not roll back with code.** They are data. To return a course
to an earlier adapter, make sure the VM still maps that version
(`./scripts/aiswe_finetuned.sh set-mapping`, then `restart`), then press
**Activate** on it in Admin → Models; the previously active version moves to
`offline`. The legacy alternative is publishing it again on Tillicum
(idempotent, and refused unless the fine-tuned service can serve that version,
exactly as activation is):

```bash
./training/promote_qlora_adapter.sh --course <courseId> --version <previousVersion> /gpfs/projects/simswe/$USER/training_outputs/qlora-runs/<courseId>/<run>-full/adapter
```

Migrations have no down scripts. Every one so far is additive — new nullable
columns, new tables, new indexes — so rolling back application code does not
require reversing them. Rolling back to a build before authentication leaves
the identity tables in place and unused; the application before migration 002
never reads them.
