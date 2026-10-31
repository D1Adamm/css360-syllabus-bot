# Reference copies of the production configuration

Sanitized copies of configuration that lives outside the repository on
`aiswe.uwb.edu`, as read on 2026-10-31
([verification-history.md](../verification-history.md#production-vm-configuration-2026-10-31)).
They exist so the next change can be diffed against what production actually
runs, and so a rebuilt host starts from the verified shape.

No script installs these. The tracked templates that *are* installed by
scripts live elsewhere: `training/inference_service/aiswe-finetuned.service`
and `ollama-finetuned.service` (via `scripts/aiswe_finetuned.sh install` and
`sudo cp`), and `scripts/systemd/aiswe-gpu-failback.*` (via
`scripts/gpu_failback_watchdog.sh install`).

| File | What it is | Where it lives on the VM |
| --- | --- | --- |
| [aiswe-backend.service](aiswe-backend.service) | The backend's systemd user unit; `@REPO_ROOT@` stands for the checkout path | `~/.config/systemd/user/aiswe-backend.service` |
| [nginx-aiswe.conf](nginx-aiswe.conf) | The Nginx site: SPA root, the `/api/` proxy block, the TLS listener; Certbot's certificate lines are left as comments and an unrelated location for another application on the host is omitted | `/etc/nginx/conf.d/aiswe.conf` |
| [backend.env.production.example](backend.env.production.example) | The keys `backend/.env` sets (or should set) on the VM, secrets as placeholders; every other key and its default is in `backend/.env.example` | `~/css360-syllabus-bot/backend/.env` (mode 600) |

When production changes, change the copy here in the same commit and add a
dated line to `verification-history.md`.
