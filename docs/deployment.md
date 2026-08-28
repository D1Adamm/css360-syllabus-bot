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
a no-op rather than an error.

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
| `CORS_ALLOWED_ORIGINS` | yes | The site origin |
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
require reversing them.
