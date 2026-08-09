#!/usr/bin/env bash
# Sync ONE course's gitignored training export to the Tillicum repo checkout.
#
# Usage (from repository root, on a machine that already has the export):
#   ./scripts/sync_training_data_to_tillicum.sh css-360-winter-2026-a7rp
#   ./scripts/sync_training_data_to_tillicum.sh css-360-winter-2026-a7rp --yes

set -euo pipefail

TILLICUM_LOGIN="${TILLICUM_LOGIN:-${USER}@tillicum.hyak.uw.edu}"
TILLICUM_REPO_ROOT="${TILLICUM_REPO_ROOT:-/gpfs/projects/simswe/${USER}/css360-syllabus-bot}"
ASSUME_YES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HELPERS="${REPO_ROOT}/scripts/lib/qlora_training_helpers.py"

usage() {
  cat <<'EOF'
Sync data/exports/<courseId>/ to the Tillicum project repo (JSONL only).

Usage:
  ./scripts/sync_training_data_to_tillicum.sh <courseId>
  ./scripts/sync_training_data_to_tillicum.sh <courseId> --yes

Environment:
  TILLICUM_LOGIN       default: $USER@tillicum.hyak.uw.edu
  TILLICUM_REPO_ROOT   default: /gpfs/projects/simswe/$USER/css360-syllabus-bot

Requires local train.jsonl, validation.jsonl, and manifest.json.
Uses rsync when available. UW Duo/password prompts remain interactive.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

helpers() {
  python3 "${HELPERS}" "$@"
}

COURSE_ID=""
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
      die "Unknown option: $1 (try --help)"
      ;;
    *)
      if [[ -n "${COURSE_ID}" ]]; then
        die "Expected exactly one course ID (extra: $1)"
      fi
      COURSE_ID="$1"
      shift
      ;;
  esac
done

[[ -n "${COURSE_ID}" ]] || die "Course ID required. Example: ./scripts/sync_training_data_to_tillicum.sh css-360-winter-2026-a7rp"
command -v python3 >/dev/null 2>&1 || die "Required command not found: python3"
command -v ssh >/dev/null 2>&1 || die "Required command not found: ssh"
[[ -f "${HELPERS}" ]] || die "Missing helpers: ${HELPERS}"

COURSE_ID="$(helpers validate-course-id "${COURSE_ID}")" || die "Invalid course ID."
LOCAL_DIR="${REPO_ROOT}/data/exports/${COURSE_ID}"
REMOTE_DIR="${TILLICUM_REPO_ROOT}/data/exports/${COURSE_ID}"

COUNTS_JSON="$(helpers validate-export-dir "${LOCAL_DIR}")" || die "Local export validation failed for ${LOCAL_DIR}"
TRAIN_COUNT="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["train_count"])' "${COUNTS_JSON}")"
VAL_COUNT="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["validation_count"])' "${COUNTS_JSON}")"

echo "Training data sync plan"
echo
echo "Local:"
echo "  ${LOCAL_DIR}/"
echo
echo "Remote:"
echo "  ${TILLICUM_LOGIN}:${REMOTE_DIR}/"
echo
echo "Train examples: ${TRAIN_COUNT}"
echo "Validation examples: ${VAL_COUNT}"
echo
echo "This syncs ONLY data/exports/${COURSE_ID}/ (not the whole repo)."
echo "No --delete is used. Unrelated remote files are left alone."
echo

if [[ "${ASSUME_YES}" -ne 1 ]]; then
  printf "Proceed with rsync/scp? [y/N] "
  read -r reply </dev/tty || die "Could not read confirmation from terminal."
  case "${reply}" in
    y|Y|yes|YES) ;;
    *)
      echo "Aborted."
      exit 1
      ;;
  esac
fi

echo
echo "Opening SSH/rsync (complete UW/Duo authentication if prompted)..."

if command -v rsync >/dev/null 2>&1; then
  # Trailing slashes: sync directory contents into the remote course export dir.
  rsync -av \
    -e ssh \
    "${LOCAL_DIR}/" \
    "${TILLICUM_LOGIN}:${REMOTE_DIR}/" \
    || die "rsync failed."
else
  echo "rsync not found; falling back to scp -r (less ideal)."
  ssh "${TILLICUM_LOGIN}" "mkdir -p '${REMOTE_DIR}'" \
    || die "Could not create remote directory via ssh."
  scp -r \
    "${LOCAL_DIR}/train.jsonl" \
    "${LOCAL_DIR}/validation.jsonl" \
    "${LOCAL_DIR}/manifest.json" \
    "${TILLICUM_LOGIN}:${REMOTE_DIR}/" \
    || die "scp failed."
fi

echo
echo "Verifying remote files..."
ssh "${TILLICUM_LOGIN}" \
  "test -s '${REMOTE_DIR}/train.jsonl' && test -s '${REMOTE_DIR}/validation.jsonl' && test -s '${REMOTE_DIR}/manifest.json'" \
  || die "Remote verification failed: required files missing under ${REMOTE_DIR}"

echo
echo "Training data sync COMPLETE"
echo "Remote export: ${REMOTE_DIR}"
echo
echo "On Tillicum run:"
echo "  cd ${TILLICUM_REPO_ROOT}"
echo "  ./training/start_qlora_training.sh --course ${COURSE_ID} --smoke"
