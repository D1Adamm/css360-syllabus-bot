# CSS 360 model evolution and evidence record

**Status:** evidence record, written 2026-09-16 from a read-only inventory of
the deployed UWB VM, the Tillicum cluster checkout, the PostgreSQL registry,
Slurm accounting, and this repository, carried out on 2026-09-15 and
2026-09-16 (PDT). The inventory changed nothing.

**Course:** `css-360-winter-2026-a7rp`.

**Production rule:** CSS 360 v2 is `course_models.current_version`, `ready`,
and `online`. This document records; it promotes nothing.

**Companion file:** [`evaluation/model_lineage.json`](../evaluation/model_lineage.json),
the machine-readable form of this record. There, every fact not established
by a direct artifact is `null` with an entry under `evidenceGaps`.

---

## 1. How to read this record

### Sources and path conventions

| Prefix | Meaning |
| --- | --- |
| `vm:` | The UWB VM `aiswe`, paths relative to the deploying account's home directory. The checkout is `vm:~/css360-syllabus-bot`; CPU training runs live under `vm:~/training_outputs/qlora-runs/css-360-winter-2026-a7rp/`; GGUF conversions under `vm:~/model_artifacts/`. |
| `tillicum:$PROJ` | The cluster project directory `/gpfs/projects/simswe/<user>`. The checkout is `tillicum:$PROJ/css360-syllabus-bot`; runs live under `tillicum:$PROJ/training_outputs/`. |
| `db:` | PostgreSQL on the VM, named by table and column. |
| `sacct:` | Slurm accounting on Tillicum, queried 2026-09-16. |
| `ollama:` | The VM's Ollama 0.32.5: `ollama list`, `ollama show <tag> --modelfile`, and `GET /api/tags`, queried 2026-09-15. |
| `repo:` | This repository at the named commit. |
| `laptop:` | The developer checkout on the author's Mac, cited only for the July 2026 split candidate. |

Run-directory files are cited by name: `runtime-report.json`,
`resolved_config.json`, `training_metrics.json`, `evaluation_metrics.json`,
`trainer_state.json`, `adapter/adapter_config.json`,
`adapter/adapter_model.safetensors`, and for VM runs `run-meta.env`.

### Timestamps

Every timestamp names its zone. Runtime reports, manifests, and the database
store UTC. Slurm accounting, file listings, Ollama, and service logs report
Pacific Daylight Time, PDT, which is UTC−7 throughout the period covered.
Where a moment matters it is written as PDT with UTC in parentheses.

### What "proven" means here

A value is proven when a file, a database row, an accounting record, or a log
line states it. A candidate is named as a candidate. Where two records agree
only in a derived quantity, the agreement is reported as agreement and not as
identity. Nothing in this record is inferred from memory.

---

## 2. Summary

| | v1 | v2 | v3 | v4-VM | v4-Tillicum |
| --- | --- | --- | --- | --- | --- |
| Role | historical | production: current, online | experimental, registered | experimental, unregistered | experimental, unregistered |
| Trained on | Tillicum, node g007, H200 | Tillicum, node g004, H200 | UWB VM, CPU | UWB VM, CPU | Tillicum, node g007, H200 |
| Started, PDT | 2026-07-21 13:07:58 | 2026-09-04 21:04:28 | 2026-09-10 16:05:37 | 2026-09-12 00:52:03 | 2026-09-15 11:18:26 |
| Job or run | Slurm 182729, legacy directory | Slurm 277072, queue run `run-20260905t040410z-c54c33` | `20260910T230535Z-full` | `20260912T075201Z-full` | Slurm 296331, `20260915T181814Z-full`, launched by hand |
| Code | `27f3b15` | `732f0fe` | `c5772f2` | `1104ad5` | `1104ad5` |
| Training data | 48 train, 6 validation, bare records; bytes unknown | 48 / 6, bare; checksums recorded, bytes not retained | 48 / 6, bare; identity with v2 unverified | 122 / 15, mixed-v4; hashes recorded | byte-identical to v4-VM |
| Window; loss | 512; full sequence | 512; full sequence | 512; full sequence | 2048; completion-only via `SFTConfig` | 2048; completion-only via collator |
| Optimizer steps | 18 | 18 | 18 | 48 | 48 |
| Saved adapter | F32, `07164b4a…` | F32, `8466d16e…` | BF16, `0a12db15…` | BF16, `1e845b5a…` | F32, `60052cf8…` |
| GGUF; Ollama tag | F16; `css360-ft-v1-test` | F32; `css360-ft-v2` | F32; `css360-cpu-v3-test` | F16; `css360-v4-test` | F32; `css360-v4-tillicum-test` |
| Registry | v1 ready, offline | v2 ready, online, current | v3 ready, offline | none | none |
| Fine-tuned service mapping | no | yes | yes | no | no |
| Trainer total | 42.36 s | 42.46 s | 3,329.9 s | 38,754.6 s | 90.65 s |
| Final eval loss | 1.6536 | 2.0056 | 1.9422 | 1.4251 | 1.4499 |

The two v4 eval losses share the same validation records and the same
intended response-only objective. They were produced by two different
completion-only implementations, `SFTConfig.completion_only_loss` on the VM
and `DataCollatorForCompletionOnlyLM` on Tillicum, which have not been shown
to produce identical label masks, so a direct numerical comparison between
them is qualified until label-mask parity is verified (section 12). Neither
is comparable with v1, v2, or v3, because v4 changed both the validation
data and the loss definition. Comparisons among v1, v2, and v3 are limited
by unverified dataset identity (section 4).

Every adapter has the same LoRA shape: rank 8, alpha 16, dropout 0.05, on
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`,
392 tensors. Every run used learning rate 2e-4, 3 epochs, per-device batch 1
with gradient accumulation 8, warmup ratio 0.1, weight decay 0.01, seed 360,
and 4-bit NF4 quantisation with double quantisation.

---

## 3. Version records

### v1: legacy Tillicum run, 2026-07-21

**Training run.** Slurm job 182729, job name `css360-qlora-train`, submitted
and started 2026-07-21 13:07:58 PDT (20:07:58 UTC), ended 13:08:52 PDT
(20:08:52 UTC), elapsed 54 s, time limit 8 h, node g007, one H200 GPU, exit 0
(`sacct:182729`). Output `tillicum:$PROJ/training_outputs/css-360-qlora/`,
the course-agnostic "live adapter path" of the pre-queue workflow. The
runtime report is stamped 2026-07-21T20:08:51Z with `gitCommitSha`
`27f3b1551d60f2c2cd2920f5b48e9bbd98dbe063` and `slurmJobId` 182729
(`runtime-report.json`). That commit is `repo:27f3b15`, "Fix Hugging Face
token path for Slurm jobs", 2026-07-21 12:53:11 PDT, fifteen minutes before
the job. The same afternoon saw two failed smokes (182721, 182723), one
completed smoke (182727, elapsed 1 min 30 s), and one comparison job
(182771), whose output is `tillicum:$PROJ/training_outputs/css-360-comparison/`
(`sacct`).

**Data.** 48 training and 6 validation records (`resolved_config.json`
`train_example_count` 48, `validation_example_count` 6;
`training/logs/train-182729.err` maps 48 and 6 examples). The job log prints
no train-file line. No hash of the files was recorded and no copy survives:
the VM holds no `train.jsonl` or `validation.jsonl` outside the current
exports (`find` on 2026-09-15), and the Tillicum export directory now holds
the 2026-09-12 v4 files. **v1's exact data bytes are unknown.** One candidate
exists: the split on the author's laptop,
`laptop:data/exports/css-360-winter-2026-a7rp/`, whose `manifest.json` records
`datasetVersion css-360-winter-2026-a7rp-approved-split-seed360-n54`,
created 2026-07-21T17:13:28Z (10:13:28 PDT) from an export at 17:07:16Z,
54 approved records split 48 and 6 at seed 360, `train.jsonl` SHA-256
`f810377f5eb5445b52003e69a8a38af09cd63ae1fc2d34c3969e5cf449b5ff5b`,
`validation.jsonl` SHA-256
`d69d1237d671f89bff7c1a61aebb209dbf1d06b63bddea4c508564f8c31062c6`. The
counts match and the split precedes the job by three hours; nothing links
those bytes to the job. It is a candidate and no more.

**Configuration.** `resolved_config.json`: `max_seq_length` 512, learning
rate 2e-4, 3 epochs, per-device batch 1, gradient accumulation 8, warmup
ratio 0.1, weight decay 0.01, seed 360, LoRA r 8, alpha 16, dropout 0.05;
`max_steps` null, so Transformers derived the 18 steps itself. The file has
no device, dtype, quantisation, or target-module fields. Those come from the
trainer source at `repo:27f3b15 training/train_qlora.py`: NF4 with double
quantisation (lines 567–568), `bnb_4bit_compute_dtype` bfloat16 when the GPU
supports it (lines 556, 569), gradient checkpointing enabled (line 578), the
chat template applied to instruction and response with no completion masking
(line 598), packing off (line 634). The loss therefore covered the whole
templated text. Warmup ran 2 of 18 steps: the learning rate is 1e-4 at step
1 and 2e-4 at step 2 (`trainer_state.json`).

**Software.** Python 3.9.25, torch 2.5.1+cu124, transformers 4.47.1,
datasets 3.2.0, peft 0.14.0, bitsandbytes 0.45.0, trl 0.13.0, accelerate
1.2.1 (`training/logs/train-182729.out`).

**Adapter.** `adapter/adapter_model.safetensors`, 48,679,352 bytes, 392
tensors, dtype F32 (safetensors header), SHA-256
`07164b4ac9a58a60551295f7ceee4ed8ce68cc1eff60ea3d76cf6cfdc55d42cc`;
`adapter/adapter_config.json` SHA-256
`a49f4b7853585275343eb49ce3e40c5af75b724bd8ab65d422f7785bac4418ad`, r 8,
alpha 16, dropout 0.05, seven targets, bias none; PEFT 0.14.0 (adapter
`README.md`). A byte-identical copy sits at
`vm:~/model_artifacts/css360-v1/adapter/` with the original file times,
2026-07-21 13:08:51 PDT.

**GGUF and Ollama.** `vm:~/model_artifacts/css360-v1/css360-v1-lora.gguf`,
24,341,024 bytes, 392 tensors, dtype F16, rank 8, alpha 16.0, written
2026-08-11 10:47:34 PDT (17:47:34 UTC), after a first `Modelfile` that
pointed at the PEFT directory (10:14:15 PDT) and before `Modelfile.gguf`
(11:13:24 PDT). Tag `css360-ft-v1-test:latest`, digest `063ce8788f88`,
2,043,734,287 bytes, modified 2026-08-11 11:13:59 PDT; Modelfile `FROM
llama3.2:3b` with `ADAPTER` on that GGUF; adapter blob
`sha256-38e976ac31acbf61f97d6397340984d759f2ef4204fc3f40417ca1d74fb5f9af`;
base blob shared with `llama3.2:3b` (`ollama`). Because the saved adapter is
F32 and the GGUF is F16, the conversion narrowed the weights; the numerical
effect of that narrowing was not measured. The tag was never mapped in the
fine-tuned service.

**Registry and serving.** `db:course_model_versions` v1: status `ready`,
deployment `offline`, `training_example_count` 54 (the approved count, as
hand-registered rows carry; see `docs/data-model.md`), `artifact_ref`
`css-360-qlora/adapter`, `run_id` NULL, `provenance` NULL, created 2026-08-10
23:22:50 PDT (2026-08-11 06:22:50 UTC), notes beginning "QLoRA adapter
trained from 54 approved examples; promoted to the live adapter path". v1 was
served by the Tillicum GPU service from the legacy path before per-course
serving existed; it is absent from
`tillicum:$PROJ/training_outputs/serving/`.

**Timings.** Queue wait 0 s (Submit equals Start) and Slurm elapsed 54 s
(`sacct`). Trainer: model load 7.01 s, training 31.35 s, evaluation 1.23 s,
total 42.36 s, 1.588 s per optimizer step, first step 2.03 s, 0.01177
GPU-hours (`runtime-report.json`). Conversion and import were not timed; the
GGUF was written at 10:47:34 and the tag at 11:13:59 PDT on 2026-08-11.

**Losses.** `train_loss` 2.6468 (`training_metrics.json`); `eval_loss`
2.5847, 1.7806, 1.6536 after epochs 1, 2, 3 (`trainer_state.json`);
`total_flos` 189,659,265,822,720. Full-sequence loss on the six validation
records of the unknown July split.

**Evidence gaps.** Data bytes; compute dtype and autocast as recorded values
(inferred from code and hardware); optimizer and update precision; Slurm
MaxRSS not queried; converter version and flags for the F16 GGUF; the
comparison job's output not read.

### v2: production, Tillicum through the training queue, 2026-09-04

**Training run.** Queue run `run-20260905t040410z-c54c33`, enqueued
2026-09-04 21:04:10 PDT (2026-09-05 04:04:10 UTC) (`db:training_runs`).
Launcher run id `20260905T040428Z`, submitted 04:04:28 UTC, wall clock
01:00:00 derived from the dataset (`training/logs/qlora-job-277072.env`).
Slurm job 277072, `qlora-train-css-360-winter-2026-a7rp`, submitted and
started 21:04:28 PDT (04:04:28 UTC), ended 21:05:21 PDT (04:05:21 UTC),
elapsed 53 s, node g004, one H200, exit 0, batch-step MaxRSS 8,021,936 KiB
(`sacct:277072`). The job's own clock read 04:04:32 UTC at start
(`training/logs/train-277072.out`). Output
`tillicum:$PROJ/training_outputs/qlora-runs/css-360-winter-2026-a7rp/20260905T040428Z-full/`.
Runtime report stamped 2026-09-05T04:05:21.208Z, `gitCommitSha`
`732f0fed8a25d869f5b281947571f92550cb843f`, which is `repo:732f0fe` "Add
final release regression coverage", 2026-09-04 00:01:48 PDT. The job
delivered its completion callback directly from the compute node
("Reported to the application.", `train-277072.out` line 106;
`training/state/pending/` is empty; `training/state/runs/run-20260905t040410z-c54c33.json`
records job 277072).

**Data.** Dataset version `css-360-winter-2026-a7rp-approved-split-seed360-n54`:
54 approved, 48 train, 6 validation. Checksums `train.jsonl`
`1dfd8259f6827eec6546af2fd6deaa14ae686ee710ef135e41a51ff94264e110` and
`validation.jsonl`
`fc5453422f1615bffd298e8e5bea7b20e3f54afd06162fce99e479bff536f6a4`
(`db:training_runs.completion.datasetChecksums` and
`db:course_model_versions.provenance.datasetChecksums`, written by the
cluster after verifying the download). The files themselves are gone: the VM
export directory was rewritten on 2026-09-12 and the Tillicum copy was
replaced by the v4 files, so both directories now carry the v4 hashes. v2's
data is identified by checksum, and its bytes are not retained. Those
checksums differ from the July laptop split's, so the two are different
files; whether their text differs is not known.

**Configuration.** `resolved_config.json`: `max_seq_length` 512, learning
rate 2e-4, 3 epochs, batch 1, accumulation 8, warmup 0.1, weight decay 0.01,
seed 360, `max_steps` 18, r 8, alpha 16, dropout 0.05; no device, dtype, or
quantisation fields. Trainer source at `repo:732f0fe`: NF4 double
quantisation (lines 646–647), compute dtype bfloat16 when supported (635,
648), gradient checkpointing (657), chat template without completion masking
(677), packing off (713). Full-sequence loss. Warmup 2 of 18
(`trainer_state.json`).

**Software.** The same venv as v1 (`train-277072.out`, "Using venv:
…/venvs/qlora"): Python 3.9.25, torch 2.5.1+cu124, transformers 4.47.1,
datasets 3.2.0, peft 0.14.0, bitsandbytes 0.45.0, trl 0.13.0, accelerate
1.2.1.

**Adapter.** 48,679,352 bytes, 392 tensors, F32, SHA-256
`8466d16e1e1be2d5fd4ec1dcd06e2ad805bc981e266bc96de39cf2255f9b7d59`;
`adapter_config.json` SHA-256
`26c99c8224f999328e93d5572d2fe517d9fe6d307c6732a02d7853068a9065b5`; PEFT
0.14.0. Byte-identical copy at `vm:~/model_artifacts/css360-v2/adapter/`,
copied 2026-09-08 01:05:16 PDT. The published copy under
`tillicum:$PROJ/training_outputs/serving/css-360-winter-2026-a7rp/v2/` was
not hashed in this inventory.

**GGUF and Ollama.** `vm:~/model_artifacts/css360-v2/css360-v2-lora.gguf`,
48,654,880 bytes, F32, 392 tensors, rank 8, written 2026-09-08 01:06:00 PDT
(08:06:00 UTC). Tag `css360-ft-v2:latest`, digest `dea74c57f25a`,
2,068,048,143 bytes, modified 2026-09-08 01:07:27 PDT; adapter blob
`sha256-af2f6c875d34141ea460ed7f4970cefc57f08ff696a692c295f5948a2cc3da41`.
Mapped in the running fine-tuned service as
`css-360-winter-2026-a7rp v2 -> css360-ft-v2:latest` (service startup log,
2026-09-12 00:14:30 PDT).

**Registry and serving.** `db:course_models.current_version` is `v2`.
`db:course_model_versions` v2: `ready`, `online`, `training_example_count`
48, `artifact_ref`
`qlora-runs/css-360-winter-2026-a7rp/20260905T040428Z-full/adapter`,
`run_id` `run-20260905t040410z-c54c33`, created 2026-09-04 21:05:21.89 PDT
(04:05:21.89 UTC), the automatic registration that also moved
`current_version` to v2; updated 21:06:43.65 PDT, the publication report.
Cluster record `tillicum:$PROJ/training_outputs/serving/css-360-winter-2026-a7rp/current.json`:
version v2, `publishedAt` 2026-09-05T04:06:43.568653Z, `sourceRef` the run's
adapter, `runId` the queue run. Stored provenance holds the commit, Slurm job
277072, dataset version and checksums, the three counts, 18 of 18 steps,
both losses, 0.011794 GPU-hours, `elapsedSeconds` 42.457, `startedAt`
2026-09-05T04:04:32Z, `completedAt` 2026-09-05T04:05:21.85Z, and a resolved
config with window 512, r 8, alpha 16. The nested runtime report and the
library versions were not part of the reporter at `732f0fe`.

**Timings.** Queue wait 0 s; Slurm elapsed 53 s (`sacct`). Trainer total
42.46 s: model load 7.81 s, training 30.65 s, evaluation 1.19 s, 1.550 s per
step, first step 1.98 s, 0.011794 GPU-hours (`runtime-report.json`).
Enqueue to registered version, 71 s. Conversion and import on 2026-09-08
(PDT): adapter copied 01:05:16, GGUF written 01:06:00, tag created 01:07:27;
the conversion itself was not timed.

**Losses.** `train_loss` 2.6118; `eval_loss` 2.8609, 2.1238, 2.0056 by epoch
(`trainer_state.json`); `total_flos` 189,149,703,045,120. Full-sequence
loss on the six validation records of the n54 split.

**Evidence gaps.** Bytes of the training files (checksums only); compute
dtype as a recorded value; optimizer and update precision; converter flags;
hash of the published serving copy.

### v3: experimental, UWB VM CPU, 2026-09-10

**Training run.** Launched by `training/start_cpu_qlora_training.sh`:
run id `20260910T230535Z`, mode full, device cpu, 6 threads, started
2026-09-10T23:05:37Z (16:05:37 PDT), log
`vm:~/css360-syllabus-bot/training/logs/cpu-train-20260910T230535Z.log`
(`run-meta.env`). Output
`vm:~/training_outputs/qlora-runs/css-360-winter-2026-a7rp/20260910T230535Z-full/`.
Runtime report stamped 2026-09-11T00:01:23.79Z (2026-09-10 17:01:23 PDT),
`gitCommitSha` `c5772f2c2977c538880daa25339b0bfad05fff1d`, `slurmJobId`
null. That commit is `repo:c5772f2`, "Fix TRL compatibility for CPU QLoRA",
2026-09-10 15:54:16 PDT. Two smokes preceded it the same afternoon: run
`20260910T215314Z` failed inside `SFTConfig(warmup_ratio=…)` on
transformers 5, the failure the commit fixed, and run `20260910T225615Z` at
`c5772f2` completed 3 steps in 231.7 s of training and 281.8 s total with
eval loss 3.809 and a full-run estimate of 1,316.5 s training-only and
2,503.3 s conservative total.

**Data.** 48 train, 6 validation (`resolved_config.json`,
`runtime-report.json`) read from
`vm:~/css360-syllabus-bot/data/exports/css-360-winter-2026-a7rp/train.jsonl`
and `validation.jsonl` as they stood on 2026-09-10 (`run-meta.env`
`TRAIN_FILE`; `resolved_config.train_path`). No checksum of that state was
recorded anywhere, and the v4 split overwrote the files on 2026-09-12. Two
observations point at v2's files without proving it: the VM export directory
shows no export between 2026-09-04 and 2026-09-12 (snapshot files
`20260904T071603Z` and `20260912T071423Z` and nothing between), and v3's
`total_flos` equals v2's, 189,149,703,045,120, which shows a compatible
total token workload. Neither is a hash. **Identity with v2's bytes is
unverified.**

**Configuration.** `resolved_config.json`: `max_seq_length` 512, learning
rate 2e-4, 3 epochs, batch 1, accumulation 8, warmup 0.1, weight decay 0.01,
seed 360, `max_steps` 18, r 8, alpha 16, dropout 0.05, device cpu,
`compute_dtype` float32, `cpu_threads` 6, gradient checkpointing true, NF4,
double quantisation true, seven targets. Full-sequence loss: no
completion-only code existed at `c5772f2`, it first appears in `1104ad5`,
and the report has no `completionOnlyLoss` field. The VM stack translated
two arguments: "max_seq_length -> max_length=512; warmup_ratio=0.1 ->
warmup_steps=2 of 18 optimizer steps" (log line 29).

**Software.** Python 3.12.13, torch 2.14.0+cu130, transformers 5.16.1,
datasets 5.0.1, peft 0.20.0, bitsandbytes 0.50.2, trl 1.12.0, accelerate
1.14.0 (`runtime-report.libraryVersions`; log).

**Adapter.** 24,365,712 bytes, 392 tensors, dtype **BF16** (safetensors
header), SHA-256
`0a12db1544e62f0ecb0971fd93ccc5214128b6dff2f49012c59b00c4c60c8d62`;
`adapter_config.json` 1,164 bytes, `peft_version` 0.20.0, r 8, alpha 16,
dropout 0.05, seven targets. The trainer requested float32 compute; the
saved dtype is a fact and its mechanism is not established (section 5).

**GGUF and Ollama.** `vm:~/model_artifacts/css360-cpu-v3-lora.gguf`,
48,655,072 bytes, F32, 392 tensors, rank 8, written 2026-09-10 17:17 PDT
(2026-09-11 00:17 UTC), F32 storage of the BF16-saved weights, whose
tensors were not compared with the adapter's; Modelfile
`css360-cpu-v3.Modelfile` 17:18 PDT. Tag `css360-cpu-v3-test:latest`, digest
`0d723f0db28f`, 2,068,048,335 bytes, modified 2026-09-10 17:18:19 PDT;
adapter blob
`sha256-63dcaa76d400fb2a57c39621f5348fdd30cadd8c2235c69c9dc88c7a980af93d`.
Mapped in the running service as v3.

**Registry.** `db:course_model_versions` v3: `ready`, `offline`,
`training_example_count` 48, `artifact_ref`
`qlora-runs/css-360-winter-2026-a7rp/20260910T230535Z-full/adapter`,
`run_id` NULL, `provenance` NULL, created 2026-09-10 22:22:19 PDT
(2026-09-11 05:22:19 UTC). `db:model_requests.updated_at` carries the same
instant, the side effect of the worker registration route, and
`current_version` stayed v2. No `admin_actions` row exists, because worker
routes do not write the audit table.

**Timings.** No queue and no Slurm. Trainer total 3,329.88 s, 55.5 min:
model load 22.64 s, training 3,252.64 s, evaluation 163.25 s including the
per-epoch evaluations, 174.36 s per step, first step 173.39 s, peak RSS
10,456.03 MiB (`runtime-report.json`; log "Total elapsed: 55.50m
(3329.9s)"). Conversion and import: GGUF 17:17, tag 17:18:19 PDT.

**Losses.** `train_loss` 2.5929; `eval_loss` 2.049 after epoch 2 (log) and
1.9422 final; `eval_mean_token_accuracy` 0.6471; `total_flos`
189,149,703,045,120. Full-sequence loss on six validation records whose
identity with v2's is unverified.

**Evidence gaps.** Dataset bytes and identity with v2; mechanism of the
BF16 save; optimizer and update precision; whether the registration passed
`setCurrent` false (inferred from the unchanged `current_version` and the
`model_requests` timestamp).

### v4-VM: experimental, UWB VM CPU, 2026-09-12

**Training run.** Launcher run `20260912T075201Z`, mode full, cpu, 6
threads, started 2026-09-12T07:52:03Z (00:52:03 PDT), log
`cpu-train-20260912T075201Z.log`, output
`vm:~/training_outputs/qlora-runs/css-360-winter-2026-a7rp/20260912T075201Z-full/`
(`run-meta.env`). Runtime report stamped 2026-09-12T18:38:17.73Z (11:38:17
PDT), `gitCommitSha` `1104ad505b2e6d18a0e97b05297b6cd602b98288`, which is
`repo:1104ad5`, "Share grounded generation and add mixed v4 training
format", 2026-09-12 00:13:29 PDT, deployed to both VM services at 00:14:30
PDT. A foreground smoke, run `20260912T072428Z` at `1104ad5`, started
07:24:30Z and completed 3 steps on a stratified 4-and-2 subset: training
1,217.7 s, total 1,328.5 s, eval loss 1.941, full-run estimate 18,637 s
training-only and 24,690.5 s conservative total, exit status 0.

**Data.** Mixed-v4 split, `datasetVersion`
`css-360-winter-2026-a7rp-mixed-split-seed360-n137`: 88 source records
rendered into 137, 122 train and 15 validation, seed 360. Composition: train
{bare/answerable 48, grounded/answerable 44, grounded/abstain 18,
grounded/false_premise 12}, validation {6, 5, 2, 2}; realised 39.4, 35.8,
14.6, 10.2 percent against targets 40, 35, 15, 10 (`manifest.json`;
`runtime-report.datasetComposition`; log line 28). Files at
`vm:~/css360-syllabus-bot/data/exports/css-360-winter-2026-a7rp/`, all
written 2026-09-12 between 00:21:45 and 00:22:24 PDT (07:21:45 to 07:22:24
UTC) and unchanged since:

| File | Lines | SHA-256 |
| --- | --- | --- |
| `approved-finetune.jsonl` | 88 | `a0dcdc240bb381c9004eea066dfff36572c931fc42f87cca4630269088bdf07f` |
| `train.jsonl` | 122 | `35818198431415964538b48c7d567a7e6449c792eb776b4d9cfce94d8ba0abd7` |
| `validation.jsonl` | 15 | `1d64373152060925b5e8af6b43221a3b6e98ccea4b9ee98035da6037f1c523c0` |
| `manifest.json` | | `f96634423cb4084f0f50c64024050ee3472ecd5c3192a525e78e0107dfd431ae` |

The manifest's own `checksums` match the first two. Its `promptTemplate`
is `grounded-v1` with fingerprint
`c963ffb5bd97aa946e730fb916b7ddedcf3a7cab9049199be186b994752d7a93`, equal to
the fingerprint of the committed template at `repo:1104ad5
backend/app/grounded_generation.py`, computed on 2026-09-16. Its
`retrieval` block records topK 4 over index SHA-256
`115a0dfca9aaa81f0fe2a2b91858bcf7108ea2f18e39d00c9fdc3f34238991fa` with
163 chunks, which is the VM's live index
(`vm:~/css360-syllabus-bot/backend/data/indexes/css-360-winter-2026-a7rp.json`,
`indexVersion` 2, created 2026-08-01T00:58:11Z). Source seeds: 54 approved
AI-generated (41 direct, 7 clarification, 6 procedure) and 34 approved
authored (20 unanswerable, 14 false-premise), the authored ones reviewed
2026-09-12 00:20:05 to 00:20:54 PDT (`db:seed_examples`;
`db:admin_actions` action `seed.review`); export summary 115 in, 88
exported, 27 skipped as pending or rejected
(`approved-export-summary.json`). Five answerable seeds were not rendered as
grounded records (`manifest.groundedSkipped`): "When and where does CSS 360
meet?" because the retrieved excerpts lacked 3:30, 5:30, 2, and 131; "Can I
use Discord without downloading an app?" and "If I know I'll miss class,
when should I notify the instructor?" because no excerpt supported the
answer; "What steps should I follow to implement a user story for Bot
Project Task #3?" missing 5; "Why do we need two screenshots for the turn-in
instead of just one?" missing 7. v4 therefore holds no grounded example of
the meeting time and place, only a bare one.

**Configuration.** `resolved_config.json`: `max_seq_length` 2048, learning
rate 2e-4, 3 epochs, batch 1, accumulation 8, warmup 0.1, weight decay 0.01,
seed 360, `max_steps` 48, r 8, alpha 16, dropout 0.05, cpu, float32, 6
threads, gradient checkpointing, NF4 double quantisation, seven targets.
Completion-only loss through `SFTConfig.completion_only_loss`
(`runtime-report.completionOnlyLoss` "config"; log line 27). Translations:
"max_seq_length -> max_length=2048; warmup_ratio=0.1 -> warmup_steps=5 of
48 optimizer steps" (log line 31).

**Software.** The same VM venv as v3.

**Adapter.** 24,365,712 bytes, 392 tensors, **BF16**, SHA-256
`1e845b5a9a1e3af37e963bb274d40b0bf77488128f435275d6366d7a666dde04`; PEFT
0.20.0; r 8, alpha 16, dropout 0.05, seven targets.

**GGUF and Ollama.** `vm:~/model_artifacts/css360-v4/css360-v4-lora.gguf`,
24,341,216 bytes, **F16**, 392 tensors, rank 8, alpha 16.0, written
2026-09-15 11:03:14 PDT (18:03:14 UTC), three days after training;
Modelfile 11:03:24 PDT. Tag `css360-v4-test:latest`, digest `b8763e820e93`,
2,043,734,479 bytes, modified 2026-09-15 11:03:34 PDT; adapter blob
`sha256-3c497e878699d8446768f4ab028869cd1b4818682f2c5d9e65b3db797b68373a`.
**Not mapped** in the fine-tuned service and **not registered**.

**Timings.** Trainer total 38,754.60 s, 10.77 h: model load 30.11 s,
training 38,297.42 s, evaluation 1,841.98 s, 772.70 s per step, first step
900.31 s, peak RSS 10,779.41 MiB (`runtime-report.json`; log "Total elapsed:
10.77h (38754.6s)"). The smoke's conservative estimate of 24,690 s undershot
by about a third, because its four-example subset is lighter than the full
mix of long grounded prompts. Conversion and import: GGUF to tag in 20 s on
2026-09-15; the conversion's start is not recorded.

**Losses.** `train_loss` 1.3616; `eval_loss` 1.4251;
`eval_mean_token_accuracy` 0.6408; `eval_num_tokens` 189,813; `total_flos`
3,224,054,650,152,960. Response-only objective through
`SFTConfig.completion_only_loss` on the 15 mixed validation records;
label-mask parity with the collator implementation used on Tillicum is
unverified.

**Evidence gaps.** Mechanism of the BF16 save; optimizer and update
precision; converter flags for the F16 GGUF; checkpoint directory not
examined.

### v4-Tillicum: experimental, Tillicum GPU, 2026-09-15

**Training run.** Launched by hand through `training/start_qlora_training.sh`
(`training/logs/qlora-job-296331.env`: `RUN_ID` `20260915T181814Z`,
`QUEUE_RUN_ID` empty, `WALLTIME` 01:00:00, `JOB_ID` 296331, `SUBMITTED_AT`
2026-09-15T18:18:22Z). Slurm job 296331,
`qlora-train-css-360-winter-2026-a7rp`, submitted 11:18:22 PDT (18:18:22
UTC), started 11:18:26 (18:18:26 UTC), ended 11:20:10 (18:20:10 UTC),
elapsed 1 min 44 s, node g007, one H200, exit 0, batch-step MaxRSS
8,042,476 KiB (`sacct:296331`). Output
`tillicum:$PROJ/training_outputs/qlora-runs/css-360-winter-2026-a7rp/20260915T181814Z-full/`.
Runtime report stamped 2026-09-15T18:20:09.57Z, `gitCommitSha` `1104ad5…`,
`slurmJobId` 296331. The cluster checkout is at `1104ad5` and clean. With no
queue run id there was no completion callback, so no `db:training_runs` row
and no registration exist.

**Data.** Byte-identical to v4-VM's inputs.
`tillicum:$PROJ/css360-syllabus-bot/data/exports/css-360-winter-2026-a7rp/`
holds `train.jsonl` at `35818198…`, `validation.jsonl` at `1d643731…`,
`manifest.json` at `f9663442…`, and `approved-finetune.jsonl` at
`a0dcdc24…`, the full values in the v4-VM table, with 122 and 15 lines and
the VM's modification times preserved to the nanosecond (`sha256sum` on the
VM on 2026-09-15 and on Tillicum on 2026-09-16). The launcher record names
these paths and the report's composition matches. The hashes are the proof;
the equal `total_flos` is corroboration and nothing more.

**Configuration.** `resolved_config.json`: `max_seq_length` 2048, learning
rate 2e-4, 3 epochs, batch 1, accumulation 8, warmup 0.1, weight decay 0.01,
seed 360, `max_steps` 48, r 8, alpha 16, dropout 0.05, device cuda,
`compute_dtype` bfloat16, gradient checkpointing true, NF4, double
quantisation true, seven targets. Completion-only loss through
`DataCollatorForCompletionOnlyLM` keyed on the Llama 3 assistant header
(`runtime-report.completionOnlyLoss` "collator"; log line 44). The learning
rate reached 2e-4 at step 5 (`trainer_state.json`).

**Software.** The same Tillicum venv as v1 and v2: Python 3.9.25, torch
2.5.1+cu124, transformers 4.47.1, datasets 3.2.0, peft 0.14.0, bitsandbytes
0.45.0, trl 0.13.0, accelerate 1.2.1 (`runtime-report.libraryVersions`;
`train-296331.out`).

**Adapter.** 48,679,352 bytes, 392 tensors, F32, SHA-256
`60052cf82890113342fa41a95edb0c91d3b58ebc77f6e7b57982e2411b79926e`;
`adapter_config.json` SHA-256
`d8e053ecf99049995c9eb98367440d16314669e2c70abd3df2f1234badfea9d7`; PEFT
0.14.0. Byte-identical copy at
`vm:~/model_artifacts/css360-v4-tillicum/adapter/`, copied 2026-09-15
22:34:32 PDT (2026-09-16 05:34:32 UTC).

**GGUF and Ollama.**
`vm:~/model_artifacts/css360-v4-tillicum/css360-v4-tillicum-lora.gguf`,
48,654,880 bytes, F32, 392 tensors, rank 8, written 2026-09-15 22:35:31 PDT
(2026-09-16 05:35:31 UTC); Modelfile 22:36:21 PDT. Tag
`css360-v4-tillicum-test:latest`, digest `d323b6f366b6`, 2,068,048,143
bytes, modified 2026-09-15 22:36:33 PDT; adapter blob
`sha256-8a958186e72738989ee198b23dfe9b90963132c2f3ed81113caf8f3b573e8528`.
**Not mapped** and **not registered**.

**Timings.** Queue wait 4 s; Slurm elapsed 104 s (`sacct`). Trainer total
90.65 s: model load 7.39 s, training 78.50 s, evaluation 4.05 s, 1.531 s per
step, first step 2.02 s, 0.025182 GPU-hours, peak RSS 5,540.45 MiB
(`runtime-report.json`; log "Total elapsed: 1.51m (90.7s)"). Conversion and
import on 2026-09-15 (PDT): adapter copied to the VM 22:34:32, GGUF written
22:35:31, tag created 22:36:33.

**Losses.** `train_loss` 1.3815; `eval_loss` 1.6263, 1.4843, 1.4499 by
epoch (`trainer_state.json`); `total_flos` 3,224,054,650,152,960. Same
validation records and intended response-only objective as v4-VM, through
`DataCollatorForCompletionOnlyLM`; label-mask parity between the two
implementations is unverified, so the two values are compared only with that
qualification.

**Evidence gaps.** Optimizer and update precision; converter flags.

---

## 4. Dataset identity

| Pair | Evidence | Verdict |
| --- | --- | --- |
| v4-VM and v4-Tillicum | Four SHA-256 values match on both hosts, with matching line counts and file times | Byte-identical, proven |
| v2 and v3 | No hash of the 2026-09-10 file state exists; the VM export history shows no export between 2026-09-04 and 2026-09-12; `total_flos` are equal | Unverified |
| v1 and the July laptop split | Counts 48 and 6, same day, three hours apart; no hash on the job side | Candidate only |
| v1 and v2 | v1's bytes are unknown; v2's checksums differ from the laptop split's; `total_flos` differ by about 0.27 percent | Unknown |

On FLOP totals: `total_flos` is the trainer's estimate of floating-point
operations. It depends on token counts, on sequence handling, and on the
accounting implementation of the library version in use. Equal totals
establish a compatible total computational and token workload; unequal
totals can arise from accounting differences between package versions as
much as from different text. Neither proves nor disproves identity of the
training text. Only the v4 pair has hash evidence.

---

## 5. Precision: four separate questions

| Artifact | 4-bit base compute dtype | Autocast | Saved PEFT adapter dtype | GGUF storage dtype | Inference base quantisation |
| --- | --- | --- | --- | --- | --- |
| v1 | bfloat16 by trainer code and H200 support; not recorded in `resolved_config.json` | bf16 by code | F32 | F16 | Q4_K_M |
| v2 | same basis as v1 | bf16 by code | F32 | F32 | Q4_K_M |
| v3 | float32 (`resolved_config.json`) | none | BF16 | F32 | Q4_K_M |
| v4-VM | float32 (`resolved_config.json`) | none | BF16 | F16 | Q4_K_M |
| v4-Tillicum | bfloat16 (`resolved_config.json`) | bf16 (`precision_settings`) | F32 | F32 | Q4_K_M |

The 4-bit base compute dtype is the `bnb_4bit_compute_dtype` the trainer
set for the dequantised base-weight operations. It does not describe every
training operation, and it says nothing about the LoRA optimizer or update
precision. The saved adapter dtype comes from the safetensors header of each
`adapter_model.safetensors`. The GGUF dtype comes from each file's tensor
table, read through llama.cpp's `gguf` library: 392 tensors, rank 8, alpha
16.0 in every file. The inference base is Ollama's `llama3.2:3b` Q4_K_M
blob `sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff`,
2,019,377,376 bytes, which every fine-tuned tag names in its `FROM` line and
which the RAG path answers from; a copy sits at
`vm:~/model_artifacts/base/llama3.2-3b-q4_k_m.gguf`.

What the saved dtype does and does not show. A BF16 adapter on disk shows
the LoRA parameters were BF16 at the moment PEFT saved them. It does not
show what dtype the optimizer states or the parameter updates used during
training. The VM trainer set float32 as the 4-bit base compute dtype and
disabled autocast, PEFT saves the live state dict at the end, and no
optimizer state or checkpoint was examined in this inventory. Optimizer and update precision is therefore **unverified
for every run**, including the Tillicum ones, whose F32 adapters likewise
say nothing about their optimizer states. One further fact narrows the VM
question: the standalone experiment `train_cpu_qlora_experiment.py`, run
on the same VM venv on 2026-09-08 with the plain Transformers `Trainer`,
saved an F32 adapter, so the BF16 result belongs to the TRL 1.12 and
transformers 5.16 path and not to the CPU.

Consequences at inference. The v1 test tag stores F32-saved weights as F16,
a narrowing whose numerical effect was not measured. v3's BF16-saved weights
are stored as F32, a widening that is exact only if the converter performed
a plain cast; the tensors were not compared. v4-VM's BF16 adapter was
converted to an F16 GGUF; the numerical effect of that conversion was not
measured. v2 and v4-Tillicum store F32-saved weights as F32; those tensors
were not compared with their adapters either. No GGUF was compared
numerically with its source adapter in this inventory.

---

## 6. Timing semantics

Eight quantities, kept apart because they measure different things.

| Quantity | Source | v1 | v2 | v3 | v4-VM | v4-Tillicum |
| --- | --- | --- | --- | --- | --- | --- |
| Queue wait, Submit to Start | `sacct` | 0 s | 0 s | none | none | 4 s |
| Slurm elapsed | `sacct` | 54 s | 53 s | none | none | 104 s |
| Trainer total process | `runtime-report.totalElapsedSeconds` | 42.36 s | 42.46 s | 3,329.88 s | 38,754.60 s | 90.65 s |
| Model load | `runtime-report.modelLoadSeconds` | 7.01 s | 7.81 s | 22.64 s | 30.11 s | 7.39 s |
| Training | `runtime-report.trainingSeconds` | 31.35 s | 30.65 s | 3,252.64 s | 38,297.42 s | 78.50 s |
| Evaluation | `runtime-report.evaluationSeconds` | 1.23 s | 1.19 s | 163.25 s | 1,841.98 s | 4.05 s |
| Seconds per optimizer step | `runtime-report.averageSecondsPerStep` | 1.588 | 1.550 | 174.36 | 772.70 | 1.531 |
| GPU-hours | `runtime-report.actualGpuHours` | 0.01177 | 0.01179 | none | none | 0.02518 |
| Conversion and import | file times only | not recorded | not recorded | not recorded | not recorded | not recorded |
| Inference latency | benchmark of 2026-09-11 | not measured | 7.6 s and 11.6 s means | 7.1 s and 11.4 s means | not measured | not measured |

Slurm elapsed exceeds the trainer's total by the job script's own work:
virtual-environment activation, `nvidia-smi`, version printing, and the
completion callback. The evaluation figures include the per-epoch
evaluations. The inference means are the service's `generationSeconds` for
the plain and grounded conditions of the historical benchmark, measured on
the deployed code of that day with an output cap of 160 tokens, so they do
not describe the current 256-token grounded path.

### The two remembered figures

**"Under ten seconds."** No complete CSS 360 training phase took under ten
seconds. The one training phase in the record that did is the smoke run of
2026-07-21, Slurm job 182727, which trained for 3.57 s over three optimizer
steps on four examples and evaluated two, with a model load of 74.58 s and
a total of 81.86 s
(`tillicum:$PROJ/training_outputs/css-360-qlora-smoke/runtime-report.json`).
A smoke run is not a model. The other values under ten seconds are model
loads of 7.0 to 7.8 s on the GPU and evaluations of 1.2 to 4.0 s. The
complete Tillicum training phases were 31.35 s for v1, 30.65 s for v2, and
78.50 s for v4-Tillicum, inside allocations of 54, 53, and 104 s.

**"A 19-minute run."** No CSS 360 training job of that length exists.
`sacct` shows three inference serving sessions of about that length, job
name `css360-ft-infer`, each cancelled by the operator: 216829 at 19 min
26 s on 2026-08-09, 267226 at 18 min 30 s on 2026-08-29, and 183404 at
17 min 44 s on 2026-07-22. The CSS 350 training jobs, a different course,
ran 42 to 53 s (253552, 262148, 264787, 265323).

### Speed ratios, each with its exact basis

| Comparison | Numerator | Denominator | Ratio |
| --- | --- | --- | --- |
| v4-VM trainer total against v4-Tillicum trainer total | 38,754.6 s | 90.65 s | about 427 |
| v4-VM training-only against v4-Tillicum training-only | 38,297.4 s | 78.50 s | about 488 |
| v4-VM trainer total against v4-Tillicum Slurm elapsed | 38,754.6 s | 104 s | about 373 |
| v3 trainer total against v2 trainer total | 3,329.9 s | 42.46 s | about 78 |
| v3 training-only against v2 training-only | 3,252.6 s | 30.65 s | about 106 |
| v3 trainer total against v2 Slurm elapsed | 3,329.9 s | 53 s | about 63 |

These are three different ratios, not one "wall-clock speedup". The trainer
totals compare like with like inside the trainer process; the training-only
ratio excludes model load and evaluation; the third mixes the VM's process
time with the cluster's allocation, which includes job-script overhead.

---

## 7. v4-VM versus v4-Tillicum

Identical: the training data bytes, the commit `1104ad5`, the seed, every
hyperparameter, the LoRA shape, the 48 optimizer steps, the 5 warmup steps,
and the dataset composition. Both adapters are applied over the same base
blob at inference.

Different: the runtime stack (Python 3.12 with torch 2.14, transformers
5.16, PEFT 0.20, TRL 1.12 against Python 3.9 with torch 2.5, transformers
4.47, PEFT 0.14, TRL 0.13); the compute path (CPU with float32 as the
4-bit base compute dtype and no autocast, against CUDA with bfloat16 as the
4-bit base compute dtype and bf16 autocast); the completion-loss
implementation (`SFTConfig.completion_only_loss` against
`DataCollatorForCompletionOnlyLM`); the saved adapter dtype (BF16 against
F32); the GGUF storage dtype (F16 against F32); and the trainer time, at the
ratios in section 6.

Therefore a behavioural difference between the two served models cannot be
attributed to hardware alone. Any comparison must name all of these as
candidate causes. On their shared validation records the two reached eval
losses of 1.4251 and 1.4499, values that share an intended response-only
objective but come from two completion-only implementations whose label-mask
parity is unverified.

---

## 8. The historical 22-question benchmark, confounded and non-controlled

Run 2026-09-11 06:12 to 06:34 UTC (2026-09-10 23:12 to 23:34 PDT) against
the deployed backend through `POST /api/model-testing/generate`, 22
questions by five configurations, 110 answers, no request failures, one
judge, one greedy pass (`evaluation/model_version_benchmark/summary.md`,
`results.json`, `scores.json`, `run.log`).

| Configuration | Overall, 1 to 5 | Hallucinations |
| --- | --- | --- |
| RAG | 3.81 | 6 of 22 |
| Fine-Tuned v2 | 1.85 | 19 of 22 |
| Fine-Tuned + RAG v2 | 3.84 | 7 of 22 |
| Fine-Tuned v3 | 1.80 | 18 of 22 |
| Fine-Tuned + RAG v3 | 4.06 | 5 of 22 |

Why it is not a controlled comparison:

- The deployed code that night carried the model-testing route
  (`repo:ce07ded`, 22:52 PDT on 2026-09-10) and predated the shared
  grounded path (`repo:1104ad5`). RAG and Fine-Tuned + RAG therefore used
  different prompt templates, different Ollama endpoints, different
  decoding, and different context windows and output caps, as the
  `grounded_generation.py` module docstring records. Their difference
  measured two pipelines, not the adapter.
- The route returned retrieved chunks but not the prompt, so prompt
  equality cannot be audited from the saved results.
- The fine-tuned output cap was 160 tokens; the current path uses 256.
- RAG timing is wall time including embedding and retrieval; fine-tuned
  timing is the service's generation time.
- Training coverage labels in `questions.json` were judged against the July
  laptop export, whose checksums differ from the v2 dataset's.
- One judge, one greedy pass, 22 questions; the v3 edge rests on 4.
- Neither v4 adapter existed.
- The RAG path has since changed, so the RAG row is not a baseline for a new
  run either.

The result stands as a historical observation about that night's deployed
pipelines and nothing more.

---

## 9. Production snapshot, 2026-09-15 PDT

- VM checkout at `1104ad5`, clean except an untracked
  `training/train_cpu_qlora_experiment.py`; `aiswe-backend` and
  `aiswe-finetuned` active since 2026-09-12 00:14:30 PDT (07:14:30 UTC).
- Fine-tuned service mapping: `css-350-spring-2026-n3h9 v2 ->
  css350-ft-v2:latest`, `css-360-winter-2026-a7rp v2 -> css360-ft-v2:latest`,
  `css-360-winter-2026-a7rp v3 -> css360-cpu-v3-test:latest`. Neither v4 tag
  is mapped.
- Registry: `current_version` v2; v2 `ready` `online`; v1 and v3 `ready`
  `offline`; no other rows. One `training_runs` row, the v2 run, state
  `succeeded`. `model_requests` status `ready`.
- Ollama 0.32.5 with eleven models, including the five CSS 360 tags above,
  `cpu-qlora-test`, `css350-ft-v2`, `llama3.2:3b`, `nomic-embed-text`,
  `qwen3:4b`, and `qwen3:8b`.
- Cluster: checkout at `1104ad5`, clean; serving tree holds v2 only;
  `training/state/pending/` empty; `.env.local` mode 600; no GGUF or
  Modelfile anywhere on the cluster, so every conversion happened on the VM.
- Retrieval index: 163 chunks, version 2, SHA-256 `115a0dfc…`, created
  2026-08-01T00:58:11Z. Syllabus text SHA-256
  `a00613cfed6d041073ff36a76813798eaecd75126cb334d14288c153ccf9b9c8`,
  identical to the repository copy.
- Nginx: `proxy_read_timeout 300` on `location /api/`; no proxy-level body
  or rate limits.

---

## 10. Excluded artifacts

- `cpu-qlora-test:latest`, built 2026-09-08 23:38:35 PDT from
  `vm:~/css360-syllabus-bot/training/train_cpu_qlora_experiment.py`, an
  untracked 144-line script (SHA-256 `07ca3405…`) with three synthetic
  chat samples, one epoch, accumulation 1, and a 128-token window, saved
  to `vm:~/model_artifacts/cpu-qlora-test/` (F32 adapter, SHA-256
  `f610787d9e08f36f32c5b49a56320c12370f4477d69dd0e164645f7ce71e2ba2`), GGUF
  F32 48,655,040 bytes. A feasibility test of CPU training, conversion, and
  loading; it contains no CSS 360 data and belongs in no comparison.
- `css350-ft-v2:latest`, another course.
- `css360-ft-v1-test:latest` is not excluded; it is v1's F16 test tag and is
  recorded under v1.

---

## 11. Open evidence gaps

1. v1's training data bytes, with the July laptop split as an unproven
   candidate.
2. v2's training file bytes, retained only as checksums; v3's identity with
   v2 unverified.
3. The mechanism by which the VM's TRL 1.12 and transformers 5.16 path saved
   BF16 adapters after float32 was set as the 4-bit base compute dtype.
4. Optimizer and update precision for every run; no optimizer state was
   examined.
5. The llama.cpp revision and `--outtype` used for each of the five GGUF
   conversions; only the results are known.
6. Slurm MaxRSS for job 182729; the legacy comparison job's output; the VM
   runs' checkpoint directories.
7. Whether v3's registration passed `setCurrent` false, inferred from the
   unchanged `current_version`.
8. The deployed commit during the 2026-09-11 benchmark, inferred from commit
   times rather than a deployment log.
9. Edits to the 54 AI-generated seeds between the July export and the
   September exports, which would explain differing checksums, were not
   traced.
10. Label-mask parity between the two completion-only implementations,
    `SFTConfig.completion_only_loss` on TRL 1.12 and
    `DataCollatorForCompletionOnlyLM` on TRL 0.13, has not been verified.
    Until it is, the two v4 eval losses are compared only with qualification.
11. No GGUF was compared numerically with its source adapter, so the effect
    of each conversion, including v4-VM's BF16 to F16, is unmeasured.

---

## 12. Prerequisites for the controlled benchmark

1. **Completion-mask parity.** Before the two v4 adapters are compared on
   eval loss, or a training-side claim rests on their losses, tokenise the
   same validation records through `SFTConfig.completion_only_loss` on the
   VM stack and through `DataCollatorForCompletionOnlyLM` on the Tillicum
   stack and compare the label arrays. Identical masks make the two losses
   directly comparable; different masks make them two different quantities.
   The served-answer benchmark does not depend on this check, since it scores
   generated text and not loss, but any statement about the two adapters'
   training outcomes does.
