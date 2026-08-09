#!/usr/bin/env bash
# Explicitly promote a versioned QLoRA adapter to the live inference path.
# Does NOT start/restart inference jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HELPERS="${REPO_ROOT}/scripts/lib/qlora_training_helpers.py"

ASSUME_YES=0
SOURCE_ADAPTER=""

usage() {
  cat <<'EOF'
Promote a completed versioned adapter to the live inference adapter path.

Usage:
  ./training/promote_qlora_adapter.sh /gpfs/projects/simswe/$USER/training_outputs/qlora-runs/<course>/<run>/adapter
  ./training/promote_qlora_adapter.sh --yes <adapter-path>

Live destination:
  /gpfs/projects/simswe/$USER/training_outputs/css-360-qlora/adapter

Before replacement, the existing live adapter (if any) is moved to:
  /gpfs/projects/simswe/$USER/training_outputs/adapter-backups/<UTC>/adapter

This script does not start or restart the inference Slurm service.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

helpers() {
  python3 "${HELPERS}" "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --yes|-y)
      ASSUME_YES=1
      shift
      ;;
    -*)
      die "Unknown option: $1"
      ;;
    *)
      if [[ -n "${SOURCE_ADAPTER}" ]]; then
        die "Expected exactly one adapter path"
      fi
      SOURCE_ADAPTER="$1"
      shift
      ;;
  esac
done

[[ -n "${SOURCE_ADAPTER}" ]] || die "Adapter source path required (try --help)"
command -v python3 >/dev/null 2>&1 || die "Required command not found: python3"
[[ -f "${HELPERS}" ]] || die "Missing helpers: ${HELPERS}"

SOURCE_ADAPTER="$(helpers validate-adapter-source "${SOURCE_ADAPTER}")" \
  || die "Adapter source validation failed."

LIVE_ADAPTER="/gpfs/projects/simswe/${USER}/training_outputs/css-360-qlora/adapter"
LIVE_PARENT="$(dirname "${LIVE_ADAPTER}")"
BACKUP_ROOT="$(python3 - <<PY
from importlib.machinery import SourceFileLoader
h = SourceFileLoader("h", "${HELPERS}").load_module()
print(h.backup_destination_dir(user="${USER}"))
PY
)"
BACKUP_ADAPTER="${BACKUP_ROOT}/adapter"
STAGING="${LIVE_PARENT}/.adapter.staging.$$"
STAGING_FAILED="${LIVE_PARENT}/.adapter.staging.failed.$$"

# Refuse promoting into itself / weird destinations.
if [[ "${SOURCE_ADAPTER}" == "${LIVE_ADAPTER}" ]]; then
  die "Source and live adapter paths are identical; nothing to promote."
fi
case "${SOURCE_ADAPTER}" in
  */training_outputs/css-360-qlora/adapter|*/training_outputs/css-360-qlora/adapter/)
    die "Refusing source that is already the live adapter path."
    ;;
esac

echo "QLoRA adapter promotion"
echo "Source:"
echo "  ${SOURCE_ADAPTER}"
echo "Destination (live inference adapter):"
echo "  ${LIVE_ADAPTER}"
if [[ -d "${LIVE_ADAPTER}" ]]; then
  echo "Existing live adapter will be backed up to:"
  echo "  ${BACKUP_ADAPTER}"
else
  echo "No existing live adapter directory found (fresh install)."
fi
echo
echo "Inference will keep using the old weights until the inference service is restarted."
echo

if [[ "${ASSUME_YES}" -ne 1 ]]; then
  printf "Type PROMOTE to replace the live adapter: "
  read -r reply </dev/tty || die "Could not read confirmation from terminal."
  [[ "${reply}" == "PROMOTE" ]] || {
    echo "Aborted (confirmation phrase not matched)."
    exit 1
  }
fi

mkdir -p "${LIVE_PARENT}"
mkdir -p "${BACKUP_ROOT}"

# Clean any leftover staging from a previous failed attempt.
rm -rf "${STAGING}" "${STAGING_FAILED}"

echo "Copying source -> staging..."
cp -a "${SOURCE_ADAPTER}" "${STAGING}" || {
  rm -rf "${STAGING}"
  die "Failed to copy source adapter into staging."
}

# Validate staging looks complete before touching live.
helpers validate-adapter-source "${STAGING}" >/dev/null \
  || {
    rm -rf "${STAGING}"
    die "Staged adapter failed validation; live adapter untouched."
  }

if [[ -d "${LIVE_ADAPTER}" ]]; then
  echo "Backing up live adapter..."
  if [[ -e "${BACKUP_ADAPTER}" ]]; then
    die "Backup destination already exists: ${BACKUP_ADAPTER}"
  fi
  mv "${LIVE_ADAPTER}" "${BACKUP_ADAPTER}" || {
    rm -rf "${STAGING}"
    die "Failed to move live adapter to backup. Staging removed; investigate ${LIVE_ADAPTER}."
  }
fi

echo "Activating staged adapter as live..."
if ! mv "${STAGING}" "${LIVE_ADAPTER}"; then
  echo "ERROR: Failed to move staging into live path." >&2
  if [[ -d "${BACKUP_ADAPTER}" && ! -d "${LIVE_ADAPTER}" ]]; then
    echo "Attempting to restore backup to live path..." >&2
    mv "${BACKUP_ADAPTER}" "${LIVE_ADAPTER}" || die "CRITICAL: restore from backup failed. Backup at ${BACKUP_ADAPTER}; staging may be at ${STAGING}"
    die "Promotion failed; previous live adapter restored from backup."
  fi
  mv "${STAGING}" "${STAGING_FAILED}" 2>/dev/null || true
  die "Promotion failed. Check ${STAGING_FAILED:-staging} and ${BACKUP_ADAPTER}."
fi

# Final validation of live path.
helpers validate-adapter-source "${LIVE_ADAPTER}" >/dev/null \
  || die "Live adapter failed validation after promotion. Backup (if any): ${BACKUP_ADAPTER}"

echo
echo "Promotion COMPLETE"
echo "Live adapter: ${LIVE_ADAPTER}"
if [[ -d "${BACKUP_ADAPTER}" ]]; then
  echo "Backup of previous live adapter: ${BACKUP_ADAPTER}"
fi
echo
echo "Next: restart the fine-tuned inference service so it loads the new adapter"
echo "(use the existing inference helpers; this script does not start GPU jobs)."
echo "  ./training/start_finetuned_service.sh"
echo "  # then on aiswe: ./scripts/start_finetuned_tunnel.sh <NODE>"
