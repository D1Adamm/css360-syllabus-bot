# CSS 360 controlled benchmark endpoint (Phase 2A)

**Status:** implemented locally on branch `research/css360-benchmark-endpoint`
on 2026-09-18. Not deployed. Nothing here is registered, promoted, published,
or served to a classroom. CSS 360 v2 remains `current_version`, `ready`, and
`online`; the production fine-tuned service and its mapping are untouched.

**Companion records:** [`css360-model-evolution.md`](css360-model-evolution.md)
and [`../evaluation/model_lineage.json`](../evaluation/model_lineage.json),
which the route reads and never writes.

---

## 1. Why a separate route and a separate service

The historical benchmark of 2026-09-11 compared two pipelines, not two sets of
weights (evolution record, section 8). The administrator's model-testing routes
now share one grounded path, but they resolve versions through the registry,
and the two v4 experiments have no registry row and must not get one by way of
a benchmark. Adding them to `FINETUNED_OLLAMA_MODELS` would make them servable
to a classroom through the production service.

So the benchmark is its own thing, end to end:

```
benchmark runner (Phase 2B)
  │  Authorization: Bearer <CSS360_BENCHMARK_TOKEN>
  ▼
backend  POST /api/research/css360/benchmark/{pair,standalone}
  │  course fixed to css-360-winter-2026-a7rp, topK fixed to 4,
  │  one retrieval, one grounded prompt (retrieve_and_prompt), hashes, lineage
  ▼
benchmark-only service  training/inference_service/benchmark_service.py
  │  127.0.0.1:9002, CSS360_BENCHMARK_SERVICE_URL, alias -> tag mapping,
  │  one decoding recipe for every alias
  ▼
the VM's Ollama  /api/chat, one user turn, the shared grounded options
```

Neither the route nor the service reads `FINETUNED_SERVICE_URL`,
`FINETUNED_OLLAMA_MODELS`, or PostgreSQL. The production paths do not read
anything introduced here.

### Aliases, not versions

| Alias | Lineage id | Ollama tag the record names | What it is |
| --- | --- | --- | --- |
| `base` | none (`baseModel`) | `llama3.2:3b` | the unmodified base model, the control |
| `v2` | `css360-v2` | `css360-ft-v2:latest` | production adapter |
| `v3` | `css360-v3` | `css360-cpu-v3-test:latest` | experimental CPU adapter |
| `v4_vm` | `css360-v4-vm` | `css360-v4-test:latest` | mixed-v4 adapter trained on the VM |
| `v4_tillicum` | `css360-v4-tillicum` | `css360-v4-tillicum-test:latest` | mixed-v4 adapter trained on Tillicum |

The aliases are fixed in code on both sides. Neither v4 experiment is called
v5, given a version, or registered; their `version` in every response is
`null` and their `role` is `experimental`. v1 is in the lineage record and is
deliberately not an alias.

---

## 2. The routes

Both answer **404** with `{"detail": "Not Found"}`, for every method and every
caller, unless the three enabling variables in section 5 are set. Both take a
JSON body with exactly two fields and refuse any other field with 422.

### `POST /api/research/css360/benchmark/pair`

One retrieval for the fixed course at the production depth, one grounded
prompt, every requested condition answered from that prompt.

```json
{"question": "When are office hours?",
 "conditions": ["rag", "ft_rag:v2", "ft_rag:v3", "ft_rag:v4_vm", "ft_rag:v4_tillicum"]}
```

`conditions` is optional; omitted means all five, in that order. Each may be
named at most once.

### `POST /api/research/css360/benchmark/standalone`

No retrieval. Every condition receives the whitespace-normalised question as
its one user turn, the base model included. The production Base wrapper
(`BASE_MODEL_SYSTEM_PROMPT`) is **not** applied, so here too the conditions
differ in the weights alone.

```json
{"question": "When are office hours?",
 "conditions": ["base", "ft:v2", "ft:v3", "ft:v4_vm", "ft:v4_tillicum"]}
```

### The response, both routes

```json
{
  "route": "pair",
  "courseId": "css-360-winter-2026-a7rp",
  "question": "When are office hours?",
  "prompt": "You are answering a student's question about their course. ...",
  "promptSha256": "…64 hex…",
  "promptTemplate": {"name": "grounded-v1", "sha256": "c963ffb5…"},
  "retrieval": {"topK": 4, "chunkCount": 4, "chunkIds": ["…"], "setSha256": "…", "facets": []},
  "retrievedChunks": [{"chunkId": "…", "section": "…", "text": "…", "score": 0.83}],
  "decoding": {"num_predict": 256, "temperature": 0, "repeat_penalty": 1.05,
               "repeat_last_n": 4096, "seed": 360, "num_ctx": 4096},
  "conditionTimeoutSeconds": 120.0,
  "lineageRecord": {"title": "CSS 360 model lineage", "schemaVersion": 1,
                    "generatedAt": "2026-09-16T07:18:22+00:00", "courseId": "css-360-winter-2026-a7rp"},
  "conditions": [
    {
      "condition": "ft_rag:v4_vm", "kind": "ft_rag", "alias": "v4_vm", "status": "ok",
      "outcome": "scorable",
      "answer": "…",
      "servedTag": "css360-v4-test:latest", "tagMatchesLineage": true,
      "servedDigest": "b8763e820e93…64 hex…", "digestMatchesLineage": true,
      "decodingMatchesSpec": true, "promptEchoMatches": true,
      "lineage": {
        "alias": "v4_vm", "lineageId": "css360-v4-vm", "version": null, "role": "experimental",
        "trainingLabel": "Experimental mixed-v4 CPU run on the UWB VM, 2026-09-12",
        "includeInControlledClaims": true,
        "expectedTag": "css360-v4-test:latest", "ollamaDigest": "b8763e820e93",
        "baseQuantization": "Q4_K_M", "huggingFaceId": null,
        "adapterSha256": "1e845b5a…", "adapterDtype": "BF16",
        "ggufBlobSha256": "3c497e87…", "ggufDtype": "F16", "ggufBytes": 24341216,
        "gitCommitSha": "1104ad505b2e6d18a0e97b05297b6cd602b98288"
      },
      "timing": {"wallSeconds": 14.2, "generationSeconds": 13.9,
                 "ollama": {"totalDurationNs": …, "loadDurationNs": …, "promptEvalCount": …,
                            "promptEvalDurationNs": …, "evalCount": …, "evalDurationNs": …,
                            "doneReason": "stop"}},
      "error": null
    },
    {
      "condition": "ft_rag:v4_tillicum", "kind": "ft_rag", "alias": "v4_tillicum", "status": "error",
      "outcome": "error",
      "answer": null, "servedTag": null, "servedDigest": null, "tagMatchesLineage": null,
      "digestMatchesLineage": null, "decodingMatchesSpec": null, "promptEchoMatches": null,
      "lineage": {"…": "…"},
      "timing": {"wallSeconds": 0.1, "generationSeconds": null, "ollama": null},
      "error": {"code": "alias_not_mapped", "message": "Alias \"v4_tillicum\" is not mapped …"}
    }
  ],
  "summary": {"attempted": 2, "scorable": 1, "failedGenerations": 0, "invalid": 0, "errors": 1},
  "requestSeconds": 71.4
}
```

For `standalone`, `prompt` is the normalised question, `promptTemplate` and
`retrieval` are `null`, and `retrievedChunks` is `[]`.

### What each field means

- `promptSha256`: SHA-256 of the UTF-8 bytes of `prompt`. Every condition in
  the response was answered from exactly those bytes: the service echoes the
  SHA-256 of what it forwarded to Ollama and the backend refuses a mismatch
  (`promptEchoMatches`, error code `prompt_integrity`).
- `retrieval.setSha256`: SHA-256 of the canonical JSON of the ordered list of
  `{chunkId, section, text}`. Scores are excluded (a float rendered two ways
  would hash two ways); order is included (the prompt renders the chunks in
  this order, so a reordered set is a different prompt).
- `promptTemplate.sha256`: `grounded_generation.prompt_template_fingerprint()`,
  the same value the v4 training manifests record, so a result can say it was
  produced under the template the adapters were trained on.
- `decoding`: `grounded_generation.grounded_options()`. The service reports the
  options it actually sent; a condition whose options differ is an error
  (`decoding_mismatch`) and its answer is discarded.
- `lineage`: a projection of the artifact's entry in `model_lineage.json`,
  built by name from a fixed field list (`LINEAGE_SUMMARY_FIELDS`). The
  record's run directories, log paths, Modelfile lines and host paths are never
  copied. `ggufBlobSha256` is Ollama's adapter blob digest: the SHA-256 of the
  GGUF file as Ollama imported it, which is the only GGUF hash the record
  holds. `adapterSha256` is the hash of the saved PEFT `adapter_model.safetensors`.
- `servedTag` / `tagMatchesLineage`: the tag the service reports having used,
  and whether it is the tag the record names for that alias (compared with the
  `:latest` default applied). A mismatch makes the condition **invalid and
  unscored**: `status` is `error` with code `tag_mismatch`, `answer` is null,
  and the served tag, the three flags, the lineage and the full timing are
  kept, so a saved result says exactly which artifact answered and how long
  it took.
- An answer with no text, empty or whitespace only, is a **failed, unscored
  generation**: `status` is `error` with code `empty_answer`, `answer` is
  null, and the timing is kept in full, including `generationSeconds` and
  Ollama's `doneReason` and token counts. A service that never answered is a
  different code (`service_*`) with no generation time, so the two cannot be
  confused.
- `servedDigest` / `digestMatchesLineage`: the manifest digest Ollama held for
  the served tag at the moment of the call, looked up by the service per
  generation, and whether the record's twelve-character digest for the alias
  is its prefix. A tag is a name and `ollama create` can put new bytes under an
  old one; the digest says which bytes answered. A digest the record does not
  name makes the condition **invalid and unscored** (`digest_mismatch`): the
  bytes under the tag are not the ones the record describes, so the answer
  cannot be attributed to the recorded artifact. The observed digest, the
  flags, the lineage and the timing are kept. `null` when the service could
  not look it up; an unknown digest is recorded as unknown, not as a mismatch.
- `outcome`: how the condition counts for a report. `scorable` is an answer to
  judge; `failed_generation` ran and produced no text (`empty_answer`);
  `invalid` produced text that cannot be attributed or compared
  (`tag_mismatch`, `digest_mismatch`, `decoding_mismatch`, `prompt_integrity`,
  `alias_mismatch`);
  `error` never produced an answer (every other code).
- `summary`: every requested condition counted once: `attempted`, and its
  split into `scorable`, `failedGenerations`, `invalid` and `errors`, which sum
  to it. A report's denominator is `attempted`, so a failed or invalid
  generation is never dropped from the count by being unscored.
- `timing.wallSeconds`: backend round trip for that condition, including the
  service hop. `generationSeconds`: the service's Ollama call, including any
  model load. `ollama.*`: Ollama's own accounting for the call, in nanoseconds
  and token counts; `doneReason: "length"` means the output cap was hit.

### Per-condition error codes

| Code | Meaning |
| --- | --- |
| `lineage_missing` | the record has no artifact for the alias; the service was not asked |
| `service_not_configured` | no `CSS360_BENCHMARK_SERVICE_URL` (cannot occur once the route is enabled) |
| `service_unavailable`, `service_timeout` | the benchmark service did not answer |
| `service_error` | the service or its Ollama answered 5xx |
| `alias_not_mapped` | the service has no tag for the alias, or the tag was never created in Ollama |
| `service_rejected` | any other 4xx from the service |
| `malformed_response` | the service body could not be read or lacked a field |
| `alias_mismatch` | the service answered for a different alias; discarded |
| `prompt_integrity` | the service's prompt hash is not the sent prompt's; discarded |
| `decoding_mismatch` | the service's options are not the shared grounded options; discarded |
| `tag_mismatch` | the service answered from a tag the record does not name for the alias; the condition is invalid and unscored, its answer discarded, its served tag and timing kept |
| `digest_mismatch` | the served tag's bytes have a digest the record does not name for the alias; invalid and unscored, answer discarded, observed digest and timing kept |
| `empty_answer` | the model produced no answer text; the generation failed and is unscored, its timing and `doneReason` kept |
| `timeout` | the condition exceeded the per-condition timeout |
| `unexpected` | anything else; the backend log has the traceback |

Every message is a fixed string. The bodies of failed upstream responses go to
the backend log in full and to the caller never.

### Whole-request refusals

| Status | When |
| --- | --- |
| 404 `{"detail":"Not Found"}` | the feature is off or incompletely configured, whatever the method |
| 411 | a body without `Content-Length` |
| 413 | `Content-Length` over 16 KiB, refused before the body is read |
| 401 + `WWW-Authenticate: Bearer` | no or wrong bearer token |
| 429 `code: auth_failures` | ten bad tokens from one client in ten minutes |
| 429 `code: rate_limited` | more accepted requests from one client than allowed per minute |
| 429 `code: busy` | a benchmark request is already running (`Retry-After: 5`) |
| 422 | a question over 1000 characters or blank after normalisation; an unknown, duplicated or empty condition list; more than five conditions; any field but `question` and `conditions` |
| 409 | the course has no usable syllabus index on this host (retrieval's 404, renamed so a 404 keeps meaning "off") |
| 503 | the lineage record cannot be read or is not the CSS 360 record; or retrieval's own Ollama embedding failed |

---

## 3. What is fixed server-side

| | Fixed to | Where |
| --- | --- | --- |
| Course | `css-360-winter-2026-a7rp` | `research_benchmark_lineage.CSS360_COURSE_ID` |
| Retrieval depth | `DEFAULT_TOP_K` (4) | `retrieval_diversity` |
| Prompt template | `grounded-v1` via `retrieve_and_prompt` | `grounded_rag`, `grounded_generation` |
| Decoding | `grounded_options()` | `grounded_generation`; the service pins `num_ctx` to 4096 and ignores `FINETUNED_NUM_CTX` |
| Output cap | 256 tokens | part of the options |
| Conditions | the two allowlists | `research_benchmark_routes.PAIR_CONDITIONS`, `STANDALONE_CONDITIONS` |
| Alias → tag | `CSS360_BENCHMARK_OLLAMA_MODELS` on the service | `benchmark_service.parse_alias_map`, validated against the allowlist at startup |

A caller can name a course, a tag, a model, a version, a path, a URL, a
template, decoding options or a depth in no request shape; every such field is
a 422 from the request model (`extra="forbid"`) on the backend and again on the
service.

---

## 4. Guarantees for a controlled comparison

1. **One retrieval per pair request.** `retrieve_and_prompt` runs once; its
   prompt string is passed to every condition. The tests assert one await and
   five identical prompt arguments.
2. **Identical prompt bytes.** The service hashes the prompt it forwards and
   the backend compares; the response carries the hash and the text.
3. **Identical decoding and output cap.** One service process, one options
   dictionary built by the production service's own function with the context
   window pinned; the backend compares the reported options to
   `grounded_options()` per condition and discards on drift. A backend test
   drives the client through the service's real ASGI app to a fake Ollama and
   asserts the options that reached Ollama equal `grounded_options()`.
4. **Identical ordered chunks.** The chunks are retrieved once; their order is
   the prompt's order and the set hash is order-sensitive.
5. **Self-describing results.** Every answer carries the lineage projection of
   the artifact that produced it, the tag actually served, and the record's
   own title and generation time, and the Ollama digest that served it. An
   answer from a tag the record does not name is never labelled with that
   alias's lineage: the condition is invalid instead.
6. **Isolation.** A condition that fails, times out or is unmapped is an error
   entry with its own timing; the others run and the request answers 200.

---

## 5. Configuration

### Backend (`backend/.env`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `CSS360_BENCHMARK_ENABLED` | off | `1`, `true`, `yes` or `on` enables; anything else is off |
| `CSS360_BENCHMARK_TOKEN` | unset | the bearer token; **at least 32 characters** or the route stays off. Generate with `openssl rand -hex 32` |
| `CSS360_BENCHMARK_SERVICE_URL` | unset | the benchmark service, `http://127.0.0.1:9002` on the VM |
| `CSS360_BENCHMARK_CONDITION_TIMEOUT_SECONDS` | 120 | per-condition timeout, clamped to 1–600 |
| `CSS360_BENCHMARK_MAX_CONCURRENT` | 1 | concurrent requests, clamped to 1–4; a request past the cap is 429 `busy`, not queued |
| `CSS360_BENCHMARK_REQUESTS_PER_MINUTE` | 30 | accepted requests per client per minute, clamped to 1–600 |

All three of the first group are required; missing any one of them means 404.
The token is a credential of its own: no session cookie is accepted on these
routes, and the token is accepted on no other route.

Set them with `backend/scripts/css360_benchmark_env.py`, which writes each key
exactly once however many times it is run, generates the token on the first
`--enable` and keeps it afterwards, never prints it, and leaves every other
line of `backend/.env` alone. `--disable` flips the flag only; `--show`
reports the state; `--rotate-token` replaces the token. Restart
`aiswe-backend` after any change.

### Benchmark service (its own process)

| Variable | Default | Meaning |
| --- | --- | --- |
| `CSS360_BENCHMARK_OLLAMA_MODELS` | empty | `alias=ollamaTag` entries, comma separated, aliases from the fixed five. Malformed or unknown entries stop the service at startup |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | the VM's Ollama (shared with the production service) |
| `BENCHMARK_INFERENCE_PORT` | `9002` | loopback port; `INFERENCE_PORT` is the production service's and is ignored |
| `CSS360_BENCHMARK_OLLAMA_TIMEOUT_SECONDS` | 120 | per-generation Ollama timeout |
| `CSS360_BENCHMARK_KEEP_ALIVE` | Ollama's default | how long a model stays resident after a request; `FINETUNED_KEEP_ALIVE` is ignored |

The service binds `127.0.0.1` only, with no host override. Its `/health` lists
every alias with `mapped`, `available` and the `digest` Ollama holds for its
tag; its `/generate` takes `{alias, prompt}` and nothing else, and answers with
the alias, the tag, the tag's current `modelDigest`, the answer, the prompt
hash, the options, the generation time and Ollama's accounting.

---

## 6. Security and privacy rules

- **Off is invisible.** While unconfigured, every request under the prefix,
  whatever its method, gets FastAPI's own 404 body from a middleware that runs
  before routing; the guard repeats the check.
- **Constant-time authentication.** `bearer_token_matches` reduces both sides
  to SHA-256 digests and compares them with `hmac.compare_digest`, so the time
  taken does not depend on the length or content of what was presented.
- **No session, no database.** The routes declare only
  `require_css360_benchmark_access`; `current_principal` is never called, no
  cookie is read, no cookie is set, `app.db` is not imported by any of the
  three modules, and the tests fail the request if a connection is opened.
- **No registry, no registration.** The modules do not import
  `course_model_resolution` or the model repositories, and the experimental
  aliases are resolved from code and the lineage file alone.
- **No internal facts in responses.** The lineage projection is an allowlist;
  response models refuse unknown fields at construction; every error message is
  a fixed string; upstream bodies stop at the backend log. The route tests scan
  every success and every refusal for host names, ports, loopback addresses,
  URLs, environment-variable names, database names, cluster and VM paths, log
  and model-file names, and the token itself.
- **Bounded work.** Body ≤ 16 KiB before reading; question ≤ 1000 characters;
  ≤ 5 conditions, none twice; one request at a time by default; per-condition
  timeout; per-client failure and request limiters keyed like the login
  limiter (the last `X-Forwarded-For` entry behind Nginx on loopback).

---

## 7. Deployment on the VM (not done; every step is the operator's)

Nothing below has been run. In order:

1. **Merge to `main` and pull `main` on the VM**, per `deployment.md`; the
   live checkout tracks `main` only.
2. **Create the benchmark service unit**, beside `aiswe-finetuned`, with its
   own mapping and port:

   ```ini
   # ~/.config/systemd/user/aiswe-benchmark.service
   [Unit]
   Description=CSS 360 benchmark-only inference (local Ollama)
   After=network.target

   [Service]
   WorkingDirectory=%h/css360-syllabus-bot/training/inference_service
   Environment=CSS360_BENCHMARK_OLLAMA_MODELS=base=llama3.2:3b,v2=css360-ft-v2:latest,v3=css360-cpu-v3-test:latest,v4_vm=css360-v4-test:latest,v4_tillicum=css360-v4-tillicum-test:latest
   Environment=BENCHMARK_INFERENCE_PORT=9002
   ExecStart=%h/css360-syllabus-bot/backend/.venv/bin/python benchmark_service.py
   Restart=on-failure

   [Install]
   WantedBy=default.target
   ```

   The five tags are the ones the lineage record names (section 1); all five
   already exist in the VM's Ollama per the production snapshot of 2026-09-15.
   `systemctl --user daemon-reload && systemctl --user enable --now aiswe-benchmark`,
   then `curl -s http://127.0.0.1:9002/health` should list all five aliases
   `available`.
3. **Configure the backend** in `backend/.env`: `CSS360_BENCHMARK_ENABLED=true`,
   `CSS360_BENCHMARK_TOKEN=<openssl rand -hex 32>`,
   `CSS360_BENCHMARK_SERVICE_URL=http://127.0.0.1:9002`. Restart
   `aiswe-backend`.
4. **Verify without generating:** an unauthenticated
   `curl -s -o /dev/null -w '%{http_code}' -X POST https://aiswe.uwb.edu/api/research/css360/benchmark/pair`
   should print `401`; with the flag off it prints `404`.
5. **Smoke one condition**, exactly as in the next subsection.
6. **Turn it off when not benchmarking** by setting the flag to `false` and
   restarting the backend; the routes return to 404.

### One-condition smoke, exactly

The condition is `ft_rag:v2` on the pair route: retrieval, the grounded
prompt, the service hop, the adapter path, the lineage label and the tag check
all run once, against the production adapter's existing tag. Nothing below
reads or writes PostgreSQL, and nothing changes what students get: the
registry, `current_version`, the production fine-tuned service and its mapping
are not touched. Prerequisite: the branch is merged and pulled on the VM and
`aiswe-backend` is running the pulled code.

1. In a second shell on the VM, run the service in the foreground with only
   the alias the smoke needs mapped:

   ```bash
   cd ~/css360-syllabus-bot/training/inference_service && CSS360_BENCHMARK_OLLAMA_MODELS="v2=css360-ft-v2:latest" ../../backend/.venv/bin/python benchmark_service.py
   ```

   Expect three lines: `CSS 360 benchmark service: 1 of 5 aliases mapped via
   http://127.0.0.1:11434`, `v2 -> css360-ft-v2:latest` with the other four
   `(unmapped)`, and `Listening on http://127.0.0.1:9002 (loopback only)`.

2. Health, in the first shell:

   ```bash
   curl -s http://127.0.0.1:9002/health
   ```

   Expect `"status":"ok"` and `"servable":["v2"]`; the `v2` row has
   `"mapped":true,"available":true`.

3. Enable the route. The script sets each key exactly once, so running it
   again, today or next month, cannot leave duplicates; the token is generated
   inside it and never printed:

   ```bash
   cd ~/css360-syllabus-bot/backend && .venv/bin/python scripts/css360_benchmark_env.py --enable && systemctl --user restart aiswe-backend && sleep 2 && curl -s http://127.0.0.1:8001/api/health
   ```

   Expect `created` or `updated` (or `unchanged` on a rerun), then
   `CSS360_BENCHMARK_ENABLED: true`, the service URL, `CSS360_BENCHMARK_TOKEN:
   set (64 characters)`, and the backend's health JSON. Then load the token
   into the shell for the requests below, from the file rather than by typing:

   ```bash
   CSS360_BENCHMARK_TOKEN="$(sed -n 's/^CSS360_BENCHMARK_TOKEN=//p' ~/css360-syllabus-bot/backend/.env | tail -1)"
   ```

4. The guard, before any generation:

   ```bash
   curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:8001/api/research/css360/benchmark/pair -H 'Content-Type: application/json' -d '{"question":"x"}'
   ```

   Expect `401`. The same with `-H 'Authorization: Bearer wrong'` is also
   `401`. A `GET` on the same path with the token is `405`, which shows the
   route now exists.

5. The one condition, on loopback so the proxy is not yet in the way:

   ```bash
   curl -s -X POST http://127.0.0.1:8001/api/research/css360/benchmark/pair -H "Authorization: Bearer $CSS360_BENCHMARK_TOKEN" -H 'Content-Type: application/json' --data-binary '{"question":"When are office hours?","conditions":["ft_rag:v2"]}' -o /tmp/smoke-ft_rag_v2.json -w '%{http_code}\n'
   ```

   Expect `200` after roughly ten to thirty seconds on the CPU host, longer if
   the model was not resident.

6. Check the saved result with the repository's checker, which reads the one
   condition and its three integrity flags, the served tag against the
   lineage tag, the recorded digest against the record, the answer, the
   timing and the attempt count:

   ```bash
   python3 ~/css360-syllabus-bot/backend/scripts/check_css360_benchmark_smoke.py /tmp/smoke-ft_rag_v2.json --condition ft_rag:v2
   ```

   Expect sixteen `PASS` lines, then `servedTag: 'css360-ft-v2:latest'`, a
   64-character `servedDigest` beginning `dea74c57f25a`, the timing and the
   answer, and exit status 0. A `doneReason` of `length` means the 256-token
   cap was hit; it passes here and is worth noting for Phase 2B.

7. Once through Nginx, saving the response and running the same checker on
   it, so the check through the proxy is the same check as on loopback and
   not only a status code:

   ```bash
   curl -s -X POST https://aiswe.uwb.edu/api/research/css360/benchmark/pair -H "Authorization: Bearer $CSS360_BENCHMARK_TOKEN" -H 'Content-Type: application/json' --data-binary '{"question":"When are office hours?","conditions":["ft_rag:v2"]}' -o /tmp/smoke-https-ft_rag_v2.json -w '%{http_code}\n'
   ```

   Expect `200`, then:

   ```bash
   python3 ~/css360-syllabus-bot/backend/scripts/check_css360_benchmark_smoke.py /tmp/smoke-https-ft_rag_v2.json --condition ft_rag:v2
   ```

   Expect the same sixteen `PASS` lines and exit status 0. This shows the
   proxy passed the `Authorization` header, its read timeout covered one
   condition, and the condition that came back through it is scorable with
   `tagMatchesLineage`, `decodingMatchesSpec` and `promptEchoMatches` all
   true.

8. Afterwards, turn the route off, restart, and confirm it is gone:

   ```bash
   cd ~/css360-syllabus-bot/backend && .venv/bin/python scripts/css360_benchmark_env.py --disable && systemctl --user restart aiswe-backend && sleep 2 && curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:8001/api/research/css360/benchmark/pair -H 'Content-Type: application/json' -d '{"question":"x"}'
   ```

   Expect `CSS360_BENCHMARK_ENABLED: false` and `404`. Stop the foreground
   service with Ctrl-C, `unset CSS360_BENCHMARK_TOKEN`, and remove the two
   saved responses under `/tmp`. The token stays in `backend/.env` for the
   next session; the flag is what turns the route on, and `--enable` next
   time reuses it.

If step 5 or 7 answers 200 but the checker prints a `FAIL`, the condition's
`error` and flags say why: `tag_mismatch` means the service's `v2` mapping
names a tag the lineage record does not; `empty_answer` means the model
produced nothing; `alias_not_mapped` means the service was started without
the `v2` entry; `digest_mismatch`, with the tag, decoding and prompt flags
all still true, means `css360-ft-v2:latest` was re-created since the
2026-09-15 inventory and the lineage record's digest must be re-verified
before any benchmark run. None of them changes anything on the host.

### Timing caveat for Phase 2B

Nginx's `proxy_read_timeout` on `location /api/` is 300 s (production
snapshot). Five conditions on a CPU host at 10–20 s each fit, but five
conditions with cold model loads, or a slow adapter, may not. The runner should
either request one or two conditions per call, or call the backend directly on
`127.0.0.1:8001` from the VM, which bypasses the proxy. The per-condition
timeout is separate from and does not extend the proxy's.

---

## 8. Decisions

Settled on review, 2026-09-18:

1. **A served-tag mismatch is an invalid, unscored condition.** If the
   service answers an alias from a tag other than the one the lineage record
   names (a re-import, a mapping typo), the condition is `status: error` with
   code `tag_mismatch`; its answer is discarded and its served tag, flags,
   lineage and timing are kept. An answer is never labelled with a lineage it
   did not come from.
2. **An empty answer is a failed, unscored generation.** `status: error` with
   code `empty_answer`, `answer` null, timing kept in full including Ollama's
   `doneReason`. It is distinguishable from a service failure, which carries
   no generation time.
3. **A digest mismatch is an invalid, unscored condition.** When the served
   tag matches but Ollama's digest for it is not the one the record names,
   the condition is `status: error` with code `digest_mismatch`; the observed
   digest, the flags, the lineage and the timing are kept and the answer is
   discarded. An unknown digest, when the service could not look one up, is
   recorded as `null` and does not invalidate.

Still open:

4. **The standalone control sends the bare question.** The production Base
   condition wraps the question in an ungrounded instruction prompt. The
   benchmark's `base` sends the same bytes as every `ft:*` condition so that
   the comparison is of weights alone. If the production wrapper is wanted as
   a further condition, it would be a new allowlisted condition, not a change
   to this one.
5. **The base model goes through the benchmark service too**, rather than
   through the backend's own Ollama client as the production RAG path does.
   This makes "same options, same call shape" true by construction within one
   process instead of by a pinned equality between two.
6. **The lineage record is read from the repository checkout at request
   time** (`evaluation/model_lineage.json`, cached per process) rather than
   copied into the backend. A missing or altered record is a 503 for the whole
   request; a missing artifact is an error for that alias only.
7. **`base` must be mapped explicitly** on the service. Nothing is served by
   default, including the base model.

---

## 9. Tests

```bash
cd backend && .venv/bin/python -m pytest -q tests/test_research_benchmark_routes.py tests/test_research_benchmark_client.py tests/test_research_benchmark_lineage.py tests/test_research_benchmark_service_contract.py
```
```bash
cd training/inference_service && ../../backend/.venv/bin/python -m pytest -q test_benchmark_service.py
```

`tests/route_classification.py` classifies both routes as `research_token`;
`test_route_auth_coverage.py` fails if either is mounted without the guard, and
`test_authorization_matrix.py` drives every browser principal through both,
expecting 404 while off and 401 without the bearer while on. The conftest
unsets every `CSS360_BENCHMARK_*` variable and resets the limiters for every
test, so the suite runs with the route off unless a test enables it.
