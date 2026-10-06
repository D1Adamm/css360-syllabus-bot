# Classroom GPU runbook

Commands only, in order. Why each step exists, and what every message means,
is in [classroom-gpu-mode.md](classroom-gpu-mode.md); section numbers below
(§) refer to it.

The example course is CSS360D, `css360d-fall-2026-q0ne`. VM commands run in
`~/css360-syllabus-bot` on `aiswe.uwb.edu`. Tillicum commands run in
`/gpfs/projects/simswe/$USER/css360-syllabus-bot`.

## 1. Before class (VM, any time earlier)

Confirm the VM is healthy and the watchdog is running:

```bash
./scripts/classroom_gpu_mode.sh status
```

Expect:
- `MODE: VM (normal)`;
- `backend ok`;
- `fine-tuned ok`;
- `timer RUNNING (enabled, active)`, with a `last check` a few seconds old.

If the watchdog says `NOT INSTALLED` or `NOT RUNNING`:

```bash
./scripts/gpu_failback_watchdog.sh run --dry-run
./scripts/gpu_failback_watchdog.sh install
```

## 2. Start Tillicum (about 30 minutes before class)

```bash
ssh $USER@tillicum.hyak.uw.edu
cd /gpfs/projects/simswe/$USER/css360-syllabus-bot && git pull origin main
SERVICE_QOS=normal ./training/start_finetuned_service.sh --hours 3
```

- Use `--hours 2` for a shorter session.
- The job must outlast the class: `start` refuses under 60 minutes left.
- Wait for `Fine-tuned service READY` and check that `css360d-fall-2026-q0ne` is listed under "Courses this session can answer for".
- **Ignore** the printed `start_finetuned_tunnel.sh` "Next" step. That is the old fallback (§4).
- `State: PD — waiting for a node with the requested resources` means the job is queued; see §8 below.

## 3. Switch AISWE to GPU (VM)

```bash
./scripts/classroom_gpu_mode.sh start \
  --course css360d-fall-2026-q0ne \
  --admin-email madamk@uw.edu
```

- **Prompts:** the admin password, then the UW password and Duo for the tunnel.
- **Success:** every line `PASS`, then `CLASSROOM GPU MODE ACTIVE: …`.
- **Any `FAIL`:** nothing was switched, and the class runs on the VM. The message names the check that failed.

## 4. Verify

```bash
./scripts/classroom_gpu_mode.sh status
```

Expect:
- `MODE: CLASSROOM GPU`;
- `Tunnel: open to <node>`;
- `GPU service: ok on <GPU>, N min left`;
- all four conditions routed `Tillicum GPU via http://127.0.0.1:9101`;
- watchdog `RUNNING`, last check `GPU mode, GPU healthy`.

The exit code is 0.

Optional, asking the backend itself:

```bash
./scripts/classroom_gpu_mode.sh status --verify --admin-email madamk@uw.edu
```

## 5. Monitor during class

```bash
./scripts/classroom_gpu_mode.sh status
curl -s http://127.0.0.1:9101/health
uptime && free -h
```

- `served` in `/health` rises with every answer.
- `status` warns `the GPU allocation ends within 15 minutes` near the end.
- On Tillicum: `./training/status_finetuned_service.sh`.

## 6. Automatic failback (nothing to do)

If the job reaches its end, the tunnel drops or the GPU fails 3 checks in a
row (15 s apart), the watchdog switches AISWE to the VM on its own and warms
the VM models.

`status` then shows:
- `MODE: VM (normal)`;
- `switchedBy: gpu-failback-watchdog`;
- `AUTOMATIC FAILBACK at …: <expired | gpu_unreachable | gpu_unhealthy> (job … on …)`.

The watchdog never cancels the job or closes the tunnel. When convenient,
run the after-class `stop` (step 7) to verify the VM, close the tunnel and
cancel the job.

Logs:

```bash
./scripts/gpu_failback_watchdog.sh status
journalctl --user -u aiswe-gpu-failback.service -n 20 --no-pager
```

## 7. After class

```bash
./scripts/classroom_gpu_mode.sh stop \
  --course css360d-fall-2026-q0ne \
  --admin-email madamk@uw.edu \
  --cancel-job
```

Expect:
- `PASS  cancelled Tillicum job <id>`;
- `PASS  tunnel closed`;
- `VM MODE ACTIVE: all four conditions are generated on this VM, and verified.`

If it prints `REMINDER: Tillicum job <id> is still running`, cancel it on
Tillicum:

```bash
scancel <id>
```

Confirm with:

```bash
./scripts/classroom_gpu_mode.sh status
```

Expect `MODE: VM (normal)`, `Tunnel: closed`, exit 0.

## 8. Emergency recovery

| Symptom | Do |
|---|---|
| Students get errors; `status` says `GPU MODE IS ON BUT THE GPU SERVICE IS UNREACHABLE` | Wait for the watchdog, or run `stop` (step 7) now. `stop` switches to the VM before asking for anything. |
| `stop` itself cannot run | `rm ~/.config/aiswe/generation-mode.json`. A missing file is VM mode. |
| Stale GPU mode pointing at an expired job | The watchdog fixes it on its next check; otherwise run `stop`. |
| Tunnel unreachable but the job is still running | `stop` (without `--cancel-job`), then `start` again. |
| `PD — Resources` on Tillicum before class | The job is queued. Teach on the VM; the job stays queued and starts later. Cancel it with `scancel <id>` if you no longer want it. |
| `MODE: VM` but you expected GPU | Read `switchedBy` / `AUTOMATIC FAILBACK` in `status` (§13). |
| `ollama …: nothing resident` in VM mode | `backend/.venv/bin/python scripts/warm_classroom_models.py --base-model llama3.2:3b --finetuned-model css360d-v1:latest --keep-alive 4h` |
| `ollama …: DOWN` | `systemctl status ollama` / `systemctl status ollama-finetuned` |
| Watchdog `NOT RUNNING` / `NOT INSTALLED` | `./scripts/gpu_failback_watchdog.sh install` |
| VM mode, but the tunnel is still open | `stop` closes it (§14). |

Watchdog commands:

```bash
./scripts/gpu_failback_watchdog.sh run --dry-run
./scripts/gpu_failback_watchdog.sh install
./scripts/gpu_failback_watchdog.sh status
./scripts/gpu_failback_watchdog.sh uninstall
```
