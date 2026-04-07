#!/usr/bin/env bash
# Transfer VCF-related files for LIHC tumour samples from filtered metadata.
#
# Modes:
#   --test   Dry run only
#   default  Perform the actual transfer
#
# Behaviour:
# - local mode: inventories SRC_DIR directly
# - remote mode: inventories files on the remote host over SSH, then matches locally
# - mirrors the source directory structure relative to SRC_DIR
#
# Destination:
#   ${PROJECT_ROOT}/data/raw/WGS_TCGA25/AtoL/VCF
#
# Examples:
#   bash scripts/transfer_lihc_vcfs.sh --test
#   bash scripts/transfer_lihc_vcfs.sh --metadata-csv data/derived/master_metadata.csv --cohort-label nafld
#   bash scripts/transfer_lihc_vcfs.sh --metadata-csv data/derived/master_metadata.csv --cohort-label nafld --required-complete-fields alcohol_status,hbv_status,hcv_status,nafld_status,obesity_class
#   bash scripts/transfer_lihc_vcfs.sh --test --ip 10.0.0.1 --port 22 --username alice --password 'secret'
#   bash scripts/transfer_lihc_vcfs.sh --ip 10.0.0.1 --port 22 --username alice --password 'secret'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

METADATA_CSV="${METADATA_CSV:-${PROJECT_ROOT}/data/derived/master_metadata.csv}"
SRC_DIR="${SRC_DIR:-/nas/Jason/Data/mutationDatabase/WGS_TCGA25/AtoL/VCF}"
DEST_DIR="${DEST_DIR:-${PROJECT_ROOT}/data/raw/WGS_TCGA25/AtoL/VCF}"
MANIFEST_DIR="${MANIFEST_DIR:-${PROJECT_ROOT}/data/derived/manifests}"
COHORT_LABEL="${COHORT_LABEL:-fibrosis}"
REQUIRED_COMPLETE_FIELDS="${REQUIRED_COMPLETE_FIELDS:-}"

REMOTE_IP=""
REMOTE_PORT=""
REMOTE_USERNAME=""
REMOTE_PASSWORD=""
TEST_MODE="false"

LOG_PREFIX="[transfer_lihc_vcfs]"

log() {
    printf '%s %s\n' "${LOG_PREFIX}" "$*"
}

usage() {
    cat <<'EOF'
Usage:
  bash scripts/transfer_lihc_vcfs.sh [--test] [--metadata-csv <path>] [--cohort-label <label>] [--required-complete-fields <csv>] [--ip <ip>] [--port <port>] [--username <name>] [--password <password>]

Options:
  --test        Dry run only. No files are copied.
  --metadata-csv  Metadata CSV path (default: data/derived/master_metadata.csv)
  --cohort-label  Label used for manifest naming (default: fibrosis)
  --required-complete-fields  Optional comma list of required non-missing metadata fields
  --ip          Remote SSH host
  --port        Remote SSH port
  --username    Remote SSH username
  --password    Remote SSH password

Notes:
  - If any remote argument is provided, all remote arguments must be provided.
  - Password-based remote access requires sshpass.
  - Files are transferred into:
      data/raw/WGS_TCGA25/AtoL/VCF
  - The folder structure relative to SRC_DIR is preserved.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)
            TEST_MODE="true"
            shift
            ;;
        --metadata-csv)
            METADATA_CSV="${2:-}"
            shift 2
            ;;
        --cohort-label)
            COHORT_LABEL="${2:-}"
            shift 2
            ;;
        --required-complete-fields)
            REQUIRED_COMPLETE_FIELDS="${2:-}"
            shift 2
            ;;
        --ip)
            REMOTE_IP="${2:-}"
            shift 2
            ;;
        --port)
            REMOTE_PORT="${2:-}"
            shift 2
            ;;
        --username)
            REMOTE_USERNAME="${2:-}"
            shift 2
            ;;
        --password)
            REMOTE_PASSWORD="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${COHORT_LABEL}" ]]; then
    echo "Error: --cohort-label cannot be blank." >&2
    exit 1
fi

REMOTE_MODE="false"
if [[ -n "${REMOTE_IP}" || -n "${REMOTE_PORT}" || -n "${REMOTE_USERNAME}" || -n "${REMOTE_PASSWORD}" ]]; then
    if [[ -z "${REMOTE_IP}" || -z "${REMOTE_PORT}" || -z "${REMOTE_USERNAME}" || -z "${REMOTE_PASSWORD}" ]]; then
        echo "Error: --ip, --port, --username, and --password must all be provided for remote transfer." >&2
        exit 1
    fi
    REMOTE_MODE="true"
fi

if [[ "${COHORT_LABEL}" == "fibrosis" ]]; then
    MANIFEST_PATH="${MANIFEST_DIR}/lihc_tumour_vcf_candidates.tsv"
else
    MANIFEST_PATH="${MANIFEST_DIR}/lihc_tumour_vcf_candidates_${COHORT_LABEL}.tsv"
fi

mkdir -p "${MANIFEST_DIR}" "${DEST_DIR}"

if [[ ! -f "${METADATA_CSV}" ]]; then
    echo "Error: metadata CSV not found: ${METADATA_CSV}" >&2
    exit 1
fi

if [[ "${REMOTE_MODE}" == "false" && ! -d "${SRC_DIR}" ]]; then
    echo "Error: source directory not found: ${SRC_DIR}" >&2
    exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

EXPECTED_FILES_TSV="${TMP_DIR}/expected_files.tsv"
SELECTED_SAMPLES_TSV="${TMP_DIR}/selected_samples.tsv"
MANIFEST_TMP_TSV="${TMP_DIR}/manifest.tsv"
RSYNC_FILES_TXT="${TMP_DIR}/rsync_files.txt"
UNMATCHED_SAMPLES_TSV="${TMP_DIR}/unmatched_samples.tsv"
REMOTE_FILE_LIST_TXT="${TMP_DIR}/remote_file_inventory.txt"
LOCAL_FILE_LIST_TXT="${TMP_DIR}/local_file_inventory.txt"
MATCH_DEBUG_TSV="${TMP_DIR}/match_debug.tsv"

log "Project root: ${PROJECT_ROOT}"
log "Metadata CSV: ${METADATA_CSV}"
log "Source dir: ${SRC_DIR}"
log "Destination dir: ${DEST_DIR}"
log "Cohort label: ${COHORT_LABEL}"
log "Required complete fields: ${REQUIRED_COMPLETE_FIELDS:-<none>}"
log "Manifest path: ${MANIFEST_PATH}"
log "Remote mode: ${REMOTE_MODE}"
log "Test mode: ${TEST_MODE}"

log "Step 1/4: selecting eligible samples and expected filenames from metadata"
python - "${METADATA_CSV}" "${SELECTED_SAMPLES_TSV}" "${EXPECTED_FILES_TSV}" "${REQUIRED_COMPLETE_FIELDS}" <<'PY'
import csv
import sys
from collections import defaultdict
from pathlib import Path

metadata_csv = Path(sys.argv[1])
selected_path = Path(sys.argv[2])
expected_path = Path(sys.argv[3])
required_complete_fields_raw = sys.argv[4]

required_complete_fields = [
    field.strip()
    for field in required_complete_fields_raw.split(",")
    if field.strip()
]
required_columns = set(required_complete_fields) | {
    "project_id",
    "primary_diagnosis",
    "tumour_sample_type",
    "tumour_sample_submitter_id",
    "tumour_sample_id",
    "file_name",
}

allowed_primary_diagnoses = {
    "hepatocellular carcinoma, nos",
    "hepatocellular carcinoma, clear cell type",
}

missing_tokens = {"", "na", "nan", "none", "null", "n/a"}

def clean_text(value):
    if value is None:
        return None
    text = value.strip()
    return text or None

def normalise(value):
    text = clean_text(value)
    if text is None:
        return None
    if text.lower() in missing_tokens:
        return None
    return text

with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise SystemExit("Error: metadata CSV has no header row")

    missing_columns = sorted(required_columns - set(reader.fieldnames))
    if missing_columns:
        raise SystemExit(f"Error: metadata CSV is missing required columns: {', '.join(missing_columns)}")

    sample_to_expected_files = defaultdict(set)
    sample_meta = {}

    total_rows = 0
    project_pass = 0
    diagnosis_pass = 0
    sample_type_pass = 0
    eligible_rows_pass = 0
    file_name_pass = 0

    for row in reader:
        total_rows += 1

        if normalise(row.get("project_id")) != "TCGA-LIHC":
            continue
        project_pass += 1

        primary_diagnosis = normalise(row.get("primary_diagnosis"))
        if primary_diagnosis is None or primary_diagnosis.lower() not in allowed_primary_diagnoses:
            continue
        diagnosis_pass += 1

        if normalise(row.get("tumour_sample_type")) != "Primary Tumor":
            continue
        sample_type_pass += 1

        if any(normalise(row.get(field)) is None for field in required_complete_fields):
            continue
        eligible_rows_pass += 1

        sample_submitter_id = clean_text(row.get("tumour_sample_submitter_id")) or ""
        sample_uuid = clean_text(row.get("tumour_sample_id")) or ""
        sample_key = sample_submitter_id or sample_uuid
        if not sample_key:
            continue

        file_name = clean_text(row.get("file_name"))
        if file_name is None:
            continue
        file_name_pass += 1

        sample_to_expected_files[sample_key].add(file_name)
        sample_meta[sample_key] = (sample_submitter_id, sample_uuid)

selected_rows = []
expected_rows = []

for sample_key in sorted(sample_to_expected_files):
    sample_submitter_id, sample_uuid = sample_meta.get(sample_key, ("", ""))
    expected_files = sorted(sample_to_expected_files[sample_key])

    selected_rows.append(
        {
            "sample_key": sample_key,
            "tumour_sample_submitter_id": sample_submitter_id,
            "tumour_sample_id": sample_uuid,
            "expected_file_count": str(len(expected_files)),
        }
    )

    for file_name in expected_files:
        expected_rows.append(
            {
                "sample_key": sample_key,
                "tumour_sample_submitter_id": sample_submitter_id,
                "tumour_sample_id": sample_uuid,
                "expected_file_name": file_name,
            }
        )

with selected_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["sample_key", "tumour_sample_submitter_id", "tumour_sample_id", "expected_file_count"],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(selected_rows)

with expected_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["sample_key", "tumour_sample_submitter_id", "tumour_sample_id", "expected_file_name"],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(expected_rows)

print(f"metadata_rows_total\t{total_rows}")
print(f"project_pass\t{project_pass}")
print(f"diagnosis_pass\t{diagnosis_pass}")
print(f"sample_type_pass\t{sample_type_pass}")
print(f"eligible_rows_pass\t{eligible_rows_pass}")
print(f"file_name_pass\t{file_name_pass}")
print(f"selected_samples\t{len(selected_rows)}")
print(f"expected_unique_file_records\t{len(expected_rows)}")
print(
    "required_complete_fields\t"
    + (",".join(required_complete_fields) if required_complete_fields else "<none>")
)
PY

log "Selected sample preview:"
if command -v column >/dev/null 2>&1; then
    head -n 15 "${SELECTED_SAMPLES_TSV}" | column -t -s $'\t'
else
    head -n 15 "${SELECTED_SAMPLES_TSV}"
fi

log "Expected filename preview:"
if command -v column >/dev/null 2>&1; then
    head -n 20 "${EXPECTED_FILES_TSV}" | column -t -s $'\t'
else
    head -n 20 "${EXPECTED_FILES_TSV}"
fi

log "Step 2/4: building source file inventory"
if [[ "${REMOTE_MODE}" == "true" ]]; then
    if ! command -v sshpass >/dev/null 2>&1; then
        echo "Error: sshpass is required for password-based remote access." >&2
        exit 1
    fi

    log "Collecting remote inventory via SSH from ${REMOTE_USERNAME}@${REMOTE_IP}:${SRC_DIR}"
    sshpass -p "${REMOTE_PASSWORD}" ssh \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -p "${REMOTE_PORT}" \
        "${REMOTE_USERNAME}@${REMOTE_IP}" \
        "find '${SRC_DIR}' -type f" > "${REMOTE_FILE_LIST_TXT}"

    inventory_count="$(wc -l < "${REMOTE_FILE_LIST_TXT}" | tr -d ' ')"
    log "Remote inventory file count: ${inventory_count}"
    log "Remote inventory preview:"
    head -n 20 "${REMOTE_FILE_LIST_TXT}" || true
else
    log "Collecting local inventory from ${SRC_DIR}"
    find "${SRC_DIR}" -type f > "${LOCAL_FILE_LIST_TXT}"
    inventory_count="$(wc -l < "${LOCAL_FILE_LIST_TXT}" | tr -d ' ')"
    log "Local inventory file count: ${inventory_count}"
    log "Local inventory preview:"
    head -n 20 "${LOCAL_FILE_LIST_TXT}" || true
fi

log "Step 3/4: matching expected filenames against discovered inventory"
python - \
    "${SRC_DIR}" \
    "${EXPECTED_FILES_TSV}" \
    "${REMOTE_MODE}" \
    "${REMOTE_FILE_LIST_TXT}" \
    "${LOCAL_FILE_LIST_TXT}" \
    "${MANIFEST_TMP_TSV}" \
    "${RSYNC_FILES_TXT}" \
    "${UNMATCHED_SAMPLES_TSV}" \
    "${MATCH_DEBUG_TSV}" <<'PY'
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

src_dir = Path(sys.argv[1])
remote_mode = sys.argv[3].lower() == "true"
inventory_path = Path(sys.argv[4] if remote_mode else sys.argv[5])
expected_path = Path(sys.argv[2])
manifest_path = Path(sys.argv[6])
rsync_files_path = Path(sys.argv[7])
unmatched_samples_path = Path(sys.argv[8])
match_debug_path = Path(sys.argv[9])

def is_allowed_vcf_related(name: str) -> bool:
    return (
        name.endswith(".vcf")
        or name.endswith(".vcf.gz")
        or name.endswith(".vcf.tbi")
        or name.endswith(".vcf.gz.tbi")
        or name.endswith(".idx")
    )

sample_to_expected = defaultdict(set)
sample_meta = {}
expected_to_samples = defaultdict(set)

with expected_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for row in reader:
        sample_key = row["sample_key"]
        sample_submitter_id = row["tumour_sample_submitter_id"]
        sample_uuid = row["tumour_sample_id"]
        expected_file_name = row["expected_file_name"]

        sample_to_expected[sample_key].add(expected_file_name)
        sample_meta[sample_key] = (sample_submitter_id, sample_uuid)
        expected_to_samples[expected_file_name].add(sample_key)

inventory_rows = []
basename_to_relpaths = defaultdict(set)
all_rel_paths = set()

with inventory_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        full_path = line.strip()
        if not full_path:
            continue
        name = os.path.basename(full_path)
        if not is_allowed_vcf_related(name):
            continue

        try:
            rel_path = os.path.relpath(full_path, str(src_dir))
        except ValueError:
            continue

        inventory_rows.append((full_path, rel_path, name))
        basename_to_relpaths[name].add(rel_path)
        all_rel_paths.add(rel_path)

matched_by_sample = defaultdict(set)
matched_rel_paths = set()
matched_expected_names = set()

for expected_name, sample_keys in expected_to_samples.items():
    direct_matches = set(basename_to_relpaths.get(expected_name, set()))
    secondary_matches = set()

    if not direct_matches and (expected_name.endswith(".tbi") or expected_name.endswith(".idx")):
        stem = expected_name[:-4]
        secondary_matches = set(basename_to_relpaths.get(stem, set()))

    all_matches = direct_matches | secondary_matches
    if all_matches:
        matched_expected_names.add(expected_name)

    rel_paths_to_add = set(all_matches)
    for rel_path in all_matches:
        # Include sidecar indices for matched VCF payloads when available.
        if rel_path.endswith(".vcf") or rel_path.endswith(".vcf.gz"):
            for sidecar_suffix in (".tbi", ".idx"):
                sidecar_path = rel_path + sidecar_suffix
                if sidecar_path in all_rel_paths:
                    rel_paths_to_add.add(sidecar_path)

    for rel_path in rel_paths_to_add:
        matched_rel_paths.add(rel_path)
        for sample_key in sample_keys:
            matched_by_sample[sample_key].add(rel_path)

with manifest_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["sample_key", "tumour_sample_submitter_id", "tumour_sample_id", "relative_path"],
        delimiter="\t",
    )
    writer.writeheader()
    for sample_key in sorted(sample_to_expected):
        sample_submitter_id, sample_uuid = sample_meta.get(sample_key, ("", ""))
        for rel_path in sorted(matched_by_sample.get(sample_key, set())):
            writer.writerow(
                {
                    "sample_key": sample_key,
                    "tumour_sample_submitter_id": sample_submitter_id,
                    "tumour_sample_id": sample_uuid,
                    "relative_path": rel_path,
                }
            )

with rsync_files_path.open("w", encoding="utf-8") as handle:
    for rel_path in sorted(matched_rel_paths):
        handle.write(rel_path + "\n")

with unmatched_samples_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["sample_key", "tumour_sample_submitter_id", "tumour_sample_id", "expected_file_count"],
        delimiter="\t",
    )
    writer.writeheader()
    for sample_key in sorted(sample_to_expected):
        if matched_by_sample.get(sample_key):
            continue
        sample_submitter_id, sample_uuid = sample_meta.get(sample_key, ("", ""))
        writer.writerow(
            {
                "sample_key": sample_key,
                "tumour_sample_submitter_id": sample_submitter_id,
                "tumour_sample_id": sample_uuid,
                "expected_file_count": str(len(sample_to_expected[sample_key])),
            }
        )

with match_debug_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["expected_file_name", "matched", "matched_relative_paths"],
        delimiter="\t",
    )
    writer.writeheader()
    for expected_name in sorted(expected_to_samples):
        rels = sorted(basename_to_relpaths.get(expected_name, set()))
        writer.writerow(
            {
                "expected_file_name": expected_name,
                "matched": "yes" if expected_name in matched_expected_names else "no",
                "matched_relative_paths": ";".join(rels),
            }
        )

print(f"inventory_allowed_files\t{len(inventory_rows)}")
print(f"expected_unique_names\t{len(expected_to_samples)}")
print(f"matched_expected_names\t{len(matched_expected_names)}")
print(f"matched_rel_paths\t{len(matched_rel_paths)}")
print(f"samples_with_matches\t{sum(1 for k in sample_to_expected if matched_by_sample.get(k))}")
print(f"samples_without_matches\t{sum(1 for k in sample_to_expected if not matched_by_sample.get(k))}")
PY

cp "${MANIFEST_TMP_TSV}" "${MANIFEST_PATH}"

log "Match debug preview:"
if command -v column >/dev/null 2>&1; then
    head -n 25 "${MATCH_DEBUG_TSV}" | column -t -s $'\t'
else
    head -n 25 "${MATCH_DEBUG_TSV}"
fi

log "Unmatched sample preview:"
if command -v column >/dev/null 2>&1; then
    head -n 20 "${UNMATCHED_SAMPLES_TSV}" | column -t -s $'\t'
else
    head -n 20 "${UNMATCHED_SAMPLES_TSV}"
fi

log "Matched relative paths preview:"
if [[ -s "${RSYNC_FILES_TXT}" ]]; then
    head -n 30 "${RSYNC_FILES_TXT}"
else
    echo "<none>"
fi

log "Step 4/4: transfer"
if [[ -s "${RSYNC_FILES_TXT}" ]]; then
    RSYNC_FLAGS="-avh"
    if [[ "${TEST_MODE}" == "true" ]]; then
        RSYNC_FLAGS="-avhn"
    fi

    if [[ "${REMOTE_MODE}" == "true" ]]; then
        if [[ "${TEST_MODE}" == "true" ]]; then
            log "Running remote rsync dry run"
        else
            log "Running remote rsync transfer"
        fi

        sshpass -p "${REMOTE_PASSWORD}" rsync ${RSYNC_FLAGS} \
            --files-from="${RSYNC_FILES_TXT}" \
            -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${REMOTE_PORT}" \
            "${REMOTE_USERNAME}@${REMOTE_IP}:${SRC_DIR}/" \
            "${DEST_DIR}/"
    else
        if [[ "${TEST_MODE}" == "true" ]]; then
            log "Running local rsync dry run"
        else
            log "Running local rsync transfer"
        fi

        rsync ${RSYNC_FLAGS} --files-from="${RSYNC_FILES_TXT}" "${SRC_DIR}/" "${DEST_DIR}/"
    fi
else
    log "No files matched; skipping transfer."
fi

selected_samples="$(tail -n +2 "${SELECTED_SAMPLES_TSV}" | wc -l | tr -d ' ')"
files_matched="$(wc -l < "${RSYNC_FILES_TXT}" | tr -d ' ')"
unmatched_samples="$(tail -n +2 "${UNMATCHED_SAMPLES_TSV}" | wc -l | tr -d ' ')"

echo
echo "Summary"
echo "- Mode: $( [[ "${TEST_MODE}" == "true" ]] && echo "test (dry run)" || echo "actual transfer" )"
echo "- Selected samples detected: ${selected_samples}"
echo "- Files matched: ${files_matched}"
echo "- Destination path: ${DEST_DIR}"
echo "- Manifest: ${MANIFEST_PATH}"
if [[ "${unmatched_samples}" -gt 0 ]]; then
    echo "- Selected samples with no matching VCF files: ${unmatched_samples}"
fi
