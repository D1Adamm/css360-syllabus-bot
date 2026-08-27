#!/usr/bin/env bash
# Publish a trained QLoRA adapter so the inference service can serve it.
# Does NOT start/restart inference jobs, and does not mark anything deployed.
#
# Per-course publishing (preferred):
#   ./training/promote_qlora_adapter.sh --course <courseId> --version v1 <adapter-path>
#
# Legacy single-adapter path (kept for the pre-course-isolation setup):
#   ./training/promote_qlora_adapter.sh <adapter-path>
#
# Why the course-scoped form exists
# ---------------------------------
# The legacy destination — training_outputs/css-360-qlora/adapter — has no
# course in it. Publishing CSS 360 there replaced whatever CSS 350 was being
# served with, and nothing about a request could detect it because a request
# carried no course either. Per-course publishing gives each course its own
# versioned directory and a `current.json` pointer, and the service resolves
# both from the course id on the request.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HELPERS="${REPO_ROOT}/scripts/lib/qlora_training_helpers.py"

ASSUME_YES=0
SOURCE_ADAPTER=""
COURSE_ID=""
VERSION=""
RUN_ID=""
SET_CURRENT=1
REPORT_PUBLICATION=1

usage() {
  cat <<'EOF'
Publish a completed versioned adapter so the inference service can serve it.

Per-course (preferred):
  ./training/promote_qlora_adapter.sh \
      --course css-350-spring-2026-n3h9 --version v1 \
      /gpfs/projects/simswe/$USER/training_outputs/qlora-runs/<course>/<run>/adapter

  Destination:
    /gpfs/projects/simswe/$USER/training_outputs/serving/<courseId>/<version>/adapter
  Pointer updated (unless --no-current):
    /gpfs/projects/simswe/$USER/training_outputs/serving/<courseId>/current.json

Legacy single-adapter (no course isolation):
  ./training/promote_qlora_adapter.sh <adapter-path>

  Destination:
    /gpfs/projects/simswe/$USER/training_outputs/css-360-qlora/adapter
  The existing live adapter is backed up to adapter-backups/<UTC>/adapter first.

Options:
  --course <courseId>   Publish for one course (enables the per-course layout)
  --version <vN>        Registered model version this adapter is
  --run-id <runId>      Training run it came from, recorded in current.json
  --no-current          Publish the version without making it the current one
  --no-report           Do not tell the application (leaves inference on the
                        previously published version)
  --yes                 Skip the confirmation prompt

Publishing does not restart the inference service.

It does report the publication to the application, after the copy has landed and
been validated, and that report is what inference resolves from. `status` and
`deployment` stay separate facts: `ready` means a usable adapter exists, and
`online` means this is the version the cluster is actually serving. Registering
a new version makes it ready; only publishing makes it served.
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
    --course)
      [[ $# -ge 2 ]] || die "--course requires a value"
      COURSE_ID="$2"
      shift 2
      ;;
    --version)
      [[ $# -ge 2 ]] || die "--version requires a value (v1, v2, …)"
      VERSION="$2"
      shift 2
      ;;
    --run-id)
      [[ $# -ge 2 ]] || die "--run-id requires a value"
      RUN_ID="$2"
      shift 2
      ;;
    --no-report)
      REPORT_PUBLICATION=0
      shift
      ;;
    --no-current)
      SET_CURRENT=0
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

if [[ -n "${COURSE_ID}" || -n "${VERSION}" ]]; then
  [[ -n "${COURSE_ID}" ]] || die "--version requires --course"
  [[ -n "${VERSION}" ]] || die "--course requires --version (v1, v2, …)"
fi

SOURCE_ADAPTER="$(helpers validate-adapter-source "${SOURCE_ADAPTER}")" \
  || die "Adapter source validation failed."

# --------------------------------------------------------------------------- #
# Per-course publishing
# --------------------------------------------------------------------------- #

if [[ -n "${COURSE_ID}" ]]; then
  COURSE_ID="$(helpers validate-course-id "${COURSE_ID}")" || die "Invalid course ID."
  COURSE_DIR="$(helpers course-serving-dir --user "${USER}" --course-id "${COURSE_ID}")" \
    || die "Could not build the course serving directory."
  DEST_ADAPTER="$(helpers course-version-adapter-dir --user "${USER}" \
    --course-id "${COURSE_ID}" --version "${VERSION}")" \
    || die "Could not build the destination adapter directory."
  DEST_VERSION_DIR="$(dirname "${DEST_ADAPTER}")"
  POINTER="${COURSE_DIR}/current.json"
  STAGING="${DEST_VERSION_DIR}.staging.$$"

  echo "QLoRA adapter publication"
  echo "Course:  ${COURSE_ID}"
  echo "Version: ${VERSION}"
  echo "Source:"
  echo "  ${SOURCE_ADAPTER}"
  echo "Destination:"
  echo "  ${DEST_ADAPTER}"
  if [[ "${SET_CURRENT}" -eq 1 ]]; then
    echo "Pointer to update:"
    echo "  ${POINTER}"
  else
    echo "Pointer: unchanged (--no-current)"
  fi
  echo
  echo "Other courses are untouched. This does not restart the inference"
  echo "service and does not mark the model deployed."
  echo

  if [[ -d "${DEST_ADAPTER}" ]]; then
    die "Version ${VERSION} is already published for ${COURSE_ID} at ${DEST_ADAPTER}.
A version is written once: a published adapter is what a recorded model version
refers to, and replacing it in place would make that record describe something
else. Publish the new adapter as the next version instead."
  fi

  if [[ "${ASSUME_YES}" -ne 1 ]]; then
    printf "Publish %s %s now? [y/N] " "${COURSE_ID}" "${VERSION}"
    read -r reply </dev/tty || die "Could not read confirmation from terminal."
    case "${reply}" in
      y|Y|yes|YES) ;;
      *)
        echo "Aborted."
        exit 1
        ;;
    esac
  fi

  mkdir -p "${COURSE_DIR}"
  rm -rf "${STAGING}"
  mkdir -p "${STAGING}"

  echo "Copying source -> staging..."
  cp -a "${SOURCE_ADAPTER}" "${STAGING}/adapter" || {
    rm -rf "${STAGING}"
    die "Failed to copy the source adapter into staging."
  }

  helpers validate-adapter-source "${STAGING}/adapter" >/dev/null || {
    rm -rf "${STAGING}"
    die "Staged adapter failed validation; nothing was published."
  }

  # Atomic within the version directory: the destination either does not exist
  # or is a complete, validated adapter. There is no moment where a service
  # could load a half-copied one.
  mv "${STAGING}" "${DEST_VERSION_DIR}" || {
    rm -rf "${STAGING}"
    die "Failed to move the staged version into place."
  }

  helpers validate-adapter-source "${DEST_ADAPTER}" >/dev/null \
    || die "Published adapter failed validation at ${DEST_ADAPTER}."

  SOURCE_REF="$(helpers relative-output-ref "${SOURCE_ADAPTER}")"

  if [[ "${SET_CURRENT}" -eq 1 ]]; then
    # Written through a temp file and renamed: a reader either sees the previous
    # pointer or the new one, never a truncated file.
    POINTER_TMP="${POINTER}.tmp.$$"
    python3 - "$POINTER_TMP" "$VERSION" "$COURSE_ID" "$RUN_ID" "$SOURCE_REF" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, version, course_id, run_id, source_ref = sys.argv[1:6]
payload = {
    "courseId": course_id,
    "version": version,
    "publishedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "sourceRef": source_ref,
}
if run_id:
    payload["runId"] = run_id
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY
    mv "${POINTER_TMP}" "${POINTER}" || die "Failed to update ${POINTER}."
    echo "Pointer updated: ${COURSE_ID} -> ${VERSION}"
  fi

  # Tell the application, and only now.
  #
  # Everything above has to have succeeded first. Inference resolves the
  # published version from PostgreSQL, so reporting before the copy landed would
  # route every question for this course at an adapter that is not there — the
  # exact outage this reporting exists to prevent, just caused from the other
  # side. A copy that fails reports nothing and the previously published version
  # goes on serving.
  #
  # A report that cannot be delivered is persisted under training/state/pending/
  # and sent by the next ./training/run_training_queue.sh --once. The reporter
  # exits 0 in that case on purpose: the publication is real, and a nonzero exit
  # here would make an operator think it had failed and run it again.
  if [[ "${SET_CURRENT}" -eq 1 && "${REPORT_PUBLICATION}" -eq 1 ]]; then
    echo
    echo "Telling the application ${COURSE_ID} is serving ${VERSION}..."
    python3 "${REPO_ROOT}/scripts/report_model_published.py" \
      --course-id "${COURSE_ID}" \
      --version "${VERSION}" \
      --source-ref "${SOURCE_REF}" || {
        echo "WARNING: could not report the publication. The adapter IS published;" >&2
        echo "the report is queued and will be sent by the next queue run." >&2
      }
  fi

  echo
  echo "Publication COMPLETE"
  echo "Course:  ${COURSE_ID}"
  echo "Version: ${VERSION}"
  echo "Adapter: ${DEST_ADAPTER}"
  echo
  echo "The running service picks this up on the next request for this course."
  echo "A version it already has loaded stays loaded under its own key, so a"
  echo "newly published version is served as soon as it is requested."
  echo
  echo "Start or check the service:"
  echo "  ./training/start_finetuned_service.sh"
  echo "  ./training/status_finetuned_service.sh"
  exit 0
fi

# --------------------------------------------------------------------------- #
# Legacy single-adapter path
#
# Kept working because an existing deployment may still be pointed at it, and
# because a migration is a thing an operator should do deliberately rather than
# discover when a script they have used for months changes destination.
# --------------------------------------------------------------------------- #

echo "NOTE: publishing to the legacy course-agnostic adapter path."
echo "      This path is shared by every course and has no version in it."
echo "      Prefer: --course <courseId> --version <vN>"
echo

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

echo "QLoRA adapter promotion (legacy path)"
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
echo "The per-course inference service does not read this path. To serve this"
echo "adapter, publish it for its course instead:"
echo "  ./training/promote_qlora_adapter.sh --course <courseId> --version <vN> ${SOURCE_ADAPTER}"
