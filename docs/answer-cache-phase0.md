# Answer cache, Phase 0: what to check and measure on the VM

Phase 0 of `feat/compare-answer-cache` changes nothing in the application. It
establishes the facts the cache design depends on. Nothing here restarts a
service, deploys code, or writes to the database.

- **Part A** is read-only and safe at any time.
- **Part B** puts the measurement script on the VM in its own folder, without
  touching the deployed checkout.
- **Part C** generates real answers, so it uses the VM's CPU. Run it outside
  class time only.

Replace `<admin email>` and `<course id>` in the commands. The scripts ask for
the administrator password without echoing it; it is never an argument.

## A. Read-only checks

1. Deployed commit, and anything uncommitted in the checkout:

```bash
cd ~/css360-syllabus-bot && git rev-parse --short HEAD && git status --short
```

2. Services up, and the generation mode (expect three `active` and `MODE: VM (normal)`):

```bash
systemctl --user is-active aiswe-backend aiswe-finetuned aiswe-gpu-failback.timer; cd ~/css360-syllabus-bot && ./scripts/classroom_gpu_mode.sh status | head -4
```

3. The backend runs one worker (expect no `--workers` in the line). The cache's
   in-flight sharing and the preparation worker both rely on this:

```bash
systemctl --user cat aiswe-backend | grep ExecStart
```

4. Whether migration 003 is applied (prints `responses` if it is, nothing if not):

```bash
cd ~/css360-syllabus-bot && (set -a && source backend/.env && set +a && psql "$DATABASE_URL" -Atc "select column_name from information_schema.columns where table_name = 'evaluations' and column_name = 'responses'")
```

5. Ollama version and model digests on both servers:

```bash
for p in 11434 11435; do echo "== :$p $(curl -s http://127.0.0.1:$p/api/version)"; curl -s http://127.0.0.1:$p/api/tags | python3 -c 'import json,sys; [print(m["name"], m["digest"][:16]) for m in sorted(json.load(sys.stdin)["models"], key=lambda m: m["name"])]'; done
```

6. Whether both servers report the same models with the same digests. The
   plan lets the backend read a course model's digest from its own Ollama
   (`:11434`) only if this prints `SAME`; otherwise the cache stays off for
   the two fine-tuned approaches until the fine-tuned service reports digests:

```bash
A="$(curl -s http://127.0.0.1:11434/api/tags | python3 -c 'import json,sys; print(sorted((m["name"], m["digest"]) for m in json.load(sys.stdin)["models"]))')"; B="$(curl -s http://127.0.0.1:11435/api/tags | python3 -c 'import json,sys; print(sorted((m["name"], m["digest"]) for m in json.load(sys.stdin)["models"]))')"; [ -n "$A" ] && [ "$A" = "$B" ] && echo "SAME models and digests on both servers" || echo "DIFFERENT or unreadable"
```

7. Where each Ollama server keeps its models (the same `OLLAMA_MODELS` path, or
   neither set, means one shared store):

```bash
systemctl show ollama ollama-finetuned -p Id -p Environment
```

8. The CPU (part of a cached answer's recorded identity):

```bash
lscpu | grep -E 'Model name|^CPU\(s\)'
```

9. Course questions by source and review state. The questions staff could
   select for preparation are the rows with origin `prototype` or
   `ai_generated`, any review state except `rejected`, and `from_student` = `f`:

```bash
cd ~/css360-syllabus-bot && (set -a && source backend/.env && set +a && psql "$DATABASE_URL" -c "select course_id, origin, lower(coalesce(nullif(btrim(review_status), ''), nullif(btrim(status), ''), 'generated')) as review, (participant_id is not null) as from_student, count(*) from seed_examples group by 1, 2, 3, 4 order by 1, 2, 3, 4")
```

10. Which courses have a fine-tuned model, and its state (a course with none
    prepares Base and RAG only):

```bash
cd ~/css360-syllabus-bot && (set -a && source backend/.env && set +a && psql "$DATABASE_URL" -c "select course_id, version, status, deployment from course_model_versions order by 1, 2")
```

## B. Put the script on the VM without changing the deployed checkout

The branch is cloned into its own folder. The deployed checkout, its services
and its data are not touched; the clone is deleted at the end.

```bash
git clone --depth 1 --branch feat/compare-answer-cache https://github.com/D1Adamm/css360-syllabus-bot.git ~/answer-cache-phase0
```

## C. Measurements (outside class time only)

Before each run, confirm nobody is using the site (expect `0`):

```bash
journalctl --user-unit aiswe-backend --since '-15 min' --no-pager -o cat | grep -c 'generation condition='
```

### C1. Repeatability and order of work

`scripts/answer_repeatability_check.py` asks course questions through the four
classroom routes, **one request at a time**. It waits whenever anyone else is
generating, stops after three failures in a row, refuses to run in GPU mode,
writes nothing to the database, and keeps no question or answer text.

A short trial first (28 generations, about 5 minutes):

```bash
cd ~/answer-cache-phase0 && ~/css360-syllabus-bot/backend/.venv/bin/python scripts/answer_repeatability_check.py --admin-email <admin email> --course-id <course id> --questions 3 --passes question,batched --out ~/repeatability-trial.json --yes
```

Then the full run (10 questions, 4 passes: 164 generations, about 30 minutes):

```bash
cd ~/answer-cache-phase0 && ~/css360-syllabus-bot/backend/.venv/bin/python scripts/answer_repeatability_check.py --admin-email <admin email> --course-id <course id> --out ~/repeatability-$(date -u +%Y%m%dT%H%M%SZ).json --yes
```

It reports two things:

- **Same answer every time?** Per approach, how many questions got the
  identical answer in every pass. Below 100% does not block the cache (the
  first stored answer becomes the one students see); it is recorded.
- **Order of work.** Seconds per answer when all four approaches are asked
  per question, against batches of five questions grouped by approach. This
  decides the preparation worker's order: if batching saves less than about
  10%, preparation goes question by question.

### C2. Baseline load test, in steps

`scripts/classroom_load_test.py` already exists and simulates students
pressing Compare together. Raise the level only after looking at the previous
result; stop at the first level where most requests are refused or time out.
Forty students is not part of the baseline unless decided separately.

```bash
cd ~/answer-cache-phase0 && ~/css360-syllabus-bot/backend/.venv/bin/python scripts/classroom_load_test.py --admin-email <admin email> --course-id <course id> --levels 1,5 --ollama-url http://127.0.0.1:11434 --ollama-url http://127.0.0.1:11435 --out ~/load-1-5-$(date -u +%Y%m%dT%H%M%SZ).json --yes
```

```bash
cd ~/answer-cache-phase0 && ~/css360-syllabus-bot/backend/.venv/bin/python scripts/classroom_load_test.py --admin-email <admin email> --course-id <course id> --levels 10 --ollama-url http://127.0.0.1:11434 --ollama-url http://127.0.0.1:11435 --out ~/load-10-$(date -u +%Y%m%dT%H%M%SZ).json --yes
```

```bash
cd ~/answer-cache-phase0 && ~/css360-syllabus-bot/backend/.venv/bin/python scripts/classroom_load_test.py --admin-email <admin email> --course-id <course id> --levels 20 --unique-questions --ollama-url http://127.0.0.1:11434 --ollama-url http://127.0.0.1:11435 --out ~/load-20-unique-$(date -u +%Y%m%dT%H%M%SZ).json --yes
```

### Afterwards

Remove the clone:

```bash
rm -rf ~/answer-cache-phase0
```

## What Phase 0 needs back

- The output of A1–A10.
- The two tables the repeatability run prints (or its JSON file).
- The table each load-test level prints.
