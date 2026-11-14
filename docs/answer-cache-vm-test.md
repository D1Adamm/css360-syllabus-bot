# Testing classroom preparation on the VM, beside production

A second, isolated instance of the application on the VM, built from the
feature checkout `~/css360-cache-test` (branch `feat/compare-answer-cache`):

| | Production | Test instance |
|---|---|---|
| Checkout | `~/css360-syllabus-bot` | `~/css360-cache-test` |
| Database | `syllabus_lab` | `syllabus_lab_cachetest` (a copy) |
| Backend | `127.0.0.1:8001`, unit `aiswe-backend` | `127.0.0.1:8011`, a plain process |
| Frontend | Nginx, `https://aiswe.uwb.edu` | `127.0.0.1:5180`, reached through an SSH tunnel |
| Syllabus files | `backend/data/indexes`, `backend/course_data` | its own copies |
| Models | Ollama `:11434` and `:11435`, fine-tuned service `:9001` | **the same ones** |

Production's checkout, `backend/.env`, database, services and Nginx are only
ever read. Nothing here restarts anything.

**The one thing shared is the models.** The test backend has its own
generation queue, so a generation it starts can run at the same moment as one
production starts, on the same CPU. Keep selections small (three to five
questions) and do not run this while a class is using the site. Before
anything that generates, check production is idle (expect `0`):

```bash
journalctl --user-unit aiswe-backend --since '-15 min' --no-pager -o cat | grep -c 'generation condition='
```

Every command that changes a database starts with a guard that refuses unless
the test checkout points at a database that is not production's.

## 0. Bring the test checkout up to date

```bash
cd ~/css360-cache-test && git pull --ff-only origin feat/compare-answer-cache && git rev-parse --short HEAD && git status --short
```

## 1. Python environment

The same interpreter production uses, and the same package versions
production has installed (only the backend's own requirements are installed,
not everything in production's environment):

```bash
mkdir -p ~/cachetest && cd ~/css360-cache-test/backend && ~/css360-syllabus-bot/backend/.venv/bin/python -m venv .venv && ~/css360-syllabus-bot/backend/.venv/bin/pip freeze | grep -E '^[A-Za-z0-9_.-]+==' > ~/cachetest/production-versions.txt && .venv/bin/pip install --quiet -r requirements.txt -c ~/cachetest/production-versions.txt && .venv/bin/python -c "import fastapi, psycopg, httpx, uvicorn; print('dependencies ok')"
```

Optional, about two minutes: the suites under the VM's own Python. Safe:
under pytest no `.env` is read and no database is opened.

```bash
cd ~/css360-cache-test && backend/.venv/bin/python -m pytest backend/tests -q 2>&1 | tail -2 && backend/.venv/bin/python -m pytest training scripts -q 2>&1 | tail -2
```

## 2. A separate database

Show where production's database is (no password is printed) and the exact
command that creates the test one:

```bash
cd ~/css360-cache-test && python3 scripts/cache_test_env.py describe --production-env ~/css360-syllabus-bot/backend/.env
```

Run the `sudo -u postgres createdb …` line it prints. It creates an empty
database owned by the application's role and touches nothing else. (If that
`sudo` is not available to you, say so; the alternative depends on the role's
privileges.)

## 3. Configure the test checkout

Writes `~/css360-cache-test/backend/.env`, readable by you only. It reuses
production's database server, role and password with the test database's name,
copies the few non-secret settings that decide which models answer, and turns
the answer cache on. Production's file is read, never written, and no secret
is shown:

```bash
cd ~/css360-cache-test && python3 scripts/cache_test_env.py write --production-env ~/css360-syllabus-bot/backend/.env && ls -l backend/.env
```

## 4. Copy the data and apply the migration

Copy production's data into the test database. `pg_dump` only reads
production; it takes a consistent snapshot without blocking anything:

```bash
cd ~/css360-cache-test && python3 scripts/cache_test_env.py guard --production-env ~/css360-syllabus-bot/backend/.env && ( set -a; source ~/css360-syllabus-bot/backend/.env; set +a; PROD_DSN="$DATABASE_URL"; set -a; source backend/.env; set +a; pg_dump "$PROD_DSN" -Fc --no-owner --no-acl | pg_restore --no-owner --no-acl -d "$DATABASE_URL" ) && echo "copy finished"
```

Apply migration 004 to the test database only, and count what is there:

```bash
cd ~/css360-cache-test && python3 scripts/cache_test_env.py guard --production-env ~/css360-syllabus-bot/backend/.env && ( set -a; source backend/.env; set +a; psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q -f backend/db/migrations/004_classroom_answer_cache.sql && psql "$DATABASE_URL" -Atc "select current_database(), (select count(*) from courses) as courses, (select count(*) from seed_examples) as questions, (select count(*) from answer_cache) as stored_answers" )
```

Confirm production's database did not change (expect `syllabus_lab|0`):

```bash
cd ~/css360-syllabus-bot && ( set -a; source backend/.env; set +a; psql "$DATABASE_URL" -Atc "select current_database(), count(*) from information_schema.tables where table_name in ('answer_cache', 'classroom_questions', 'classroom_preparation')" )
```

Copy the syllabus indexes (needed for RAG) and the uploaded syllabi. A copy,
so nothing the test instance does can touch production's files:

```bash
mkdir -p ~/css360-cache-test/backend/data && cp -a ~/css360-syllabus-bot/backend/data/indexes ~/css360-cache-test/backend/data/ && ls ~/css360-cache-test/backend/data/indexes | head -5; cp -a ~/css360-syllabus-bot/backend/course_data ~/css360-cache-test/backend/ 2>/dev/null; cd ~/css360-cache-test && git status --short
```

`git status` should print nothing: all of this is ignored by Git.

## 5. Start the two processes

Check the two ports are free (expect `free`):

```bash
ss -ltn | grep -E ':(8011|5180)\b' || echo free
```

Backend, on loopback port 8011:

```bash
cd ~/css360-cache-test/backend && nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8011 > ~/cachetest/backend.log 2>&1 & echo $! > ~/cachetest/backend.pid; sleep 5; curl -s http://127.0.0.1:8011/api/health; echo; grep -iE 'traceback|error|startup complete' ~/cachetest/backend.log | tail -3
```

Frontend: build it, then serve the build on loopback port 5180 with `/api`
sent to the test backend:

```bash
cd ~/css360-cache-test && npm ci 2>&1 | tail -1 && VITE_API_BASE_URL=/api npm run build 2>&1 | grep -E 'built in|error'
```

```bash
cd ~/css360-cache-test && DEV_API_PROXY_TARGET=http://127.0.0.1:8011 nohup npx vite preview --host 127.0.0.1 --port 5180 --strictPort > ~/cachetest/frontend.log 2>&1 & sleep 4; curl -s http://127.0.0.1:5180/api/health; echo
```

Both `curl`s should print `{"status":"ok","service":"syllabus-model-lab-backend"}`.

## 6. Sign in

On your own computer, open a tunnel to the VM (replace the address with the
one you normally SSH to) and leave it running:

```bash
ssh -N -L 5180:127.0.0.1:5180 <you>@<vm address>
```

Then open `http://localhost:5180/login` and sign in with your usual
administrator account: the test database is a copy, so the same email and
password work. The address is `localhost`, so its cookies are separate from
the real site's and signing in or out here does not affect your production
session.

Admin → Courses → a course → **Record** → **Classroom answers** opens the
page. In Professor view it is the **Classroom answers** row on the course
overview.

## 7. What to try

Watch the test backend's log in a second SSH window. Each model run is one
`generation condition=…` line; a stored answer produces none:

```bash
tail -f ~/cachetest/backend.log | grep --line-buffered -E 'generation condition=|classroom preparation|answer cache|answer identity'
```

1. **Select.** Tick three to five questions. The summary shows the counts at
   once; refresh the page and the selection is still there.
2. **Prepare.** Press **Prepare answers**. The log shows a
   `classroom preparation: … started` line, then one generation at a time.
   It starts at once unless a question was asked on Compare in the last 20
   seconds (the quiet period here; production's default is 120). The counts
   rise without a refresh.
3. **Stop mid-way.** Press **Stop preparation** while it is running. The run
   ends after the current answer; the page says the completed answers are
   saved and offers **Prepare remaining answers**.
4. **Use a partly prepared question.** Admin → the course → **Preview Student
   Experience** → Compare. Ask a question whose indicators show some ✓ and
   some ○ (type it with different capitals if you like). The ✓ approaches fill
   in order without a log line; the ○ ones generate and log.
5. **Resume.** Back on Classroom answers, press **Prepare remaining answers**.
   Only the missing answers are generated; it ends on **Ready**.
6. **Fully prepared.** Ask a fully prepared question on Compare: four cards
   fill about 0.4 s apart, with no `generation` line in the log.
7. **Not prepared.** Ask something nobody selected: four generations, as
   today.
8. **Interrupted run.** Tick three more questions, start preparing, and kill
   the backend while it runs:

   ```bash
   kill -9 "$(cat ~/cachetest/backend.pid)"
   ```

   Start it again with the backend command from step 5 and reload the page: it
   reports that preparation was interrupted, keeps every saved answer, and
   offers **Prepare remaining answers**.
9. **Identical questions together.** Five simulated students asking the same
   question at once should cost at most four generations (one per approach),
   not twenty. Count the log lines before and after; the script asks for your
   password without showing it:

   ```bash
   grep -c 'generation condition=' ~/cachetest/backend.log; cd ~/css360-cache-test && backend/.venv/bin/python scripts/classroom_load_test.py --base-url http://127.0.0.1:8011 --admin-email <admin email> --course-id <course id> --levels 5 --same-question --no-system --out ~/cachetest/same-question.json --yes; grep -c 'generation condition=' ~/cachetest/backend.log
   ```

What was stored, per approach, with the model that produced it (test
database; read-only):

```bash
cd ~/css360-cache-test && ( set -a; source backend/.env; set +a; psql "$DATABASE_URL" -c "select course_id, condition, source, count(*) as answers, min(identity->'model'->>'tag') as model, min(left(identity->'model'->>'digest', 12)) as digest, min(identity->'model'->>'courseVersion') as course_version from answer_cache group by 1, 2, 3 order by 1, 2, 3" )
```

## 8. The fine-tuned approaches

Use a course whose model is activated (Admin → Models).

First, what the fine-tuned service reports about itself. The cache needs
`ok`, an Ollama address, and one line per course version with its
`ollamaModel`:

```bash
curl -s http://127.0.0.1:9001/health | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("status"), d.get("ollamaUrl")); [print(m) for m in d.get("models", [])]'
```

Then all four approaches, through the test backend, with each course's
activated version (the existing production check, pointed at port 8011; it
asks for your password without showing it):

```bash
cd ~/css360-cache-test && backend/.venv/bin/python scripts/verify_finetuned_production.py --backend-url http://127.0.0.1:8011 --admin-email <admin email> --course <course id>
```

Then on the Classroom answers page for that course:

- the four indicators per question read Base, RAG, FT, FT+RAG with ○ (not –),
  and there is no notice about the fine-tuned model;
- **Prepare answers** produces `condition=fineTuned` and
  `condition=fineTunedRag` lines in the log, and the page ends on **Ready**
  with four ✓ per question;
- the query at the end of step 7 shows `fineTuned` and `fineTunedRag` rows
  with the course's Ollama model, a digest and the course version;
- on Compare, a prepared question fills all four cards from the store.

If the page instead says the course's fine-tuned model is unavailable while
the service is healthy, send the output of the `/health` command above and
these lines:

```bash
grep -E 'answer identity|answer cache' ~/cachetest/backend.log | tail -10
```

## 9. Stopping and removing the test instance

Stop the two processes (production is not involved):

```bash
kill "$(cat ~/cachetest/backend.pid)" 2>/dev/null; pkill -f 'vite preview --host 127.0.0.1 --port 5180'; sleep 1; ss -ltn | grep -E ':(8011|5180)\b' || echo stopped
```

When testing is over, the test database, the checkout and `~/cachetest` can
be removed. These delete things, so they are yours to run when you are sure:
`sudo -u postgres dropdb syllabus_lab_cachetest`, then remove
`~/css360-cache-test` and `~/cachetest`.

## What to send back

- Anything that printed an error, with the command.
- The output of the stored-answers query after preparing a course with a
  fine-tuned model.
- The two counts from "Identical questions together".
- How long a generation takes there: `grep 'generation condition=' ~/cachetest/backend.log | tail -8`.
