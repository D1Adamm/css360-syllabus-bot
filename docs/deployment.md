# Deployment

How a change reaches the UWB VM and the Tillicum checkout.

Everything here is run by hand. There is no CI, no automated deploy, and nothing
in this repository triggers a deployment.

Values below reflect the deployed setup as it is represented in this repository —
the systemd unit name and backend port come from
`scripts/start_finetuned_tunnel.sh`, the Nginx document root and the SELinux
relabel from the publish step that has been used against the VM. Anything not
represented anywhere in the repository is not documented here.

---

## The deployed shape

| Piece | Where |
| --- | --- |
| Static frontend | Nginx, document root `/usr/share/nginx/html/` |
| Backend | uvicorn on `127.0.0.1:8001`, systemd **user** unit `aiswe-backend` |
| Reverse proxy | Nginx forwards `location /api/` to the backend |
| Database | PostgreSQL on the VM, DSN in `backend/.env` |
| Generation | Ollama on the VM |
| Repository | `~/css360-syllabus-bot` on the VM, `/gpfs/projects/simswe/$USER/css360-syllabus-bot` on Tillicum |

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

---

## UWB VM

```bash
cd ~/css360-syllabus-bot && git pull origin main
```

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

```bash
systemctl --user restart aiswe-backend
```
```bash
curl -s http://127.0.0.1:8001/api/health
```

Expect `{"status":"ok","service":"syllabus-model-lab-backend"}`.

A user unit, not a system one — so it is managed without root, and it needs
lingering enabled for the account if it is to survive logout.

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
must have, checked with `sudo nginx -T`.

- `try_files $uri /index.html` (or equivalent) must already be in place for
  the SPA; `/join`, `/join/<code>`, `/invite/<token>` and `/login` are
  frontend routes served by `index.html`.
- **Forward the client address, or the whole class shares one throttle.**
  Join-code and login failures are limited per client. Uvicorn (0.49, its
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
  }
  ```

  Without the `X-Forwarded-For` line every browser reaches the backend as
  `127.0.0.1`, and ten mistyped codes anywhere in the room lock the join page
  for everyone for ten minutes. If a load balancer ever sits in front of the
  VM, add `set_real_ip_from <its address>; real_ip_header X-Forwarded-For;` so
  `$remote_addr` is the student's address rather than the balancer's.
- Nothing rewrites cookies. The backend sets `Secure; HttpOnly; SameSite=Lax;
  Path=/api` itself, from configuration rather than from proxy headers.

### Tests on the VM

Safe to run. The suite cannot reach the production database: under pytest,
`backend/.env` is not read and `DATABASE_URL` is ignored outright.

```bash
backend/.venv/bin/python -m pytest backend/tests -q
```

That barrier exists because it was once not true — see
[tillicum-operations.md](tillicum-operations.md#test-isolation) and
`backend/tests/test_test_isolation.py`.

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
| `FINETUNED_SERVICE_URL` | for fine-tuned paths | Set by the tunnel script to `http://127.0.0.1:9001` |
| `CORS_ALLOWED_ORIGINS` | yes | The site origin. Also the CSRF origin allowlist |
| `APP_PUBLIC_ORIGIN` | recommended | The site origin, for the bootstrap script's printed link |
| `AUTH_COOKIE_SECURE` | no | Default true. Never set false on the VM |
| `AUTH_*` lifetimes | no | Session and invitation lifetimes; defaults documented in `backend/.env.example` |
| `OLLAMA_*` | yes | Base and RAG generation |

**UWB VM** — `.env.local` for the frontend build: `VITE_API_BASE_URL` must be the
site origin **plus `/api`**.

**Tillicum** — `.env.local`, from `.env.example`: `TRAINING_API_BASE_URL` and
`TRAINING_WORKER_TOKEN`, matching the backend's value.

---

## Rolling back

Application code:

```bash
git -C ~/css360-syllabus-bot checkout <previous-commit> && npm ci && npm run build
```

then republish and restart as above.

**Model versions do not roll back with code.** They are data. To return a course
to an earlier adapter, publish that version again on Tillicum — publication is
idempotent and moves the previously published version to `offline`:

```bash
./training/promote_qlora_adapter.sh --course <courseId> --version <previousVersion> /gpfs/projects/simswe/$USER/training_outputs/qlora-runs/<courseId>/<run>-full/adapter
```

Migrations have no down scripts. Every one so far is additive — new nullable
columns, new tables, new indexes — so rolling back application code does not
require reversing them. Rolling back to a build before authentication leaves
the identity tables in place and unused; the application before migration 002
never reads them.
