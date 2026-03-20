#!/usr/bin/env bash
# download_datasets.sh — Download public Maxwell HD-MEA datasets for validation.
#
# Usage:
#   bash scripts/download_datasets.sh [OPTIONS]
#
# Options:
#   --data-dir DIR   Target directory (default: ./data/external/)
#   --all            Also download the very large optional datasets
#                    (DANDI 001603 — 322 GB, DANDI 001747 — 1.6 TB)
#   --dry-run        Print what would be downloaded without actually downloading
#   --help           Show this help message
#
# Notes:
#   - dandi download supports resuming interrupted downloads by default.
#     If a download is interrupted, simply re-run this script.
#   - Make this file executable:  chmod +x scripts/download_datasets.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DATA_DIR="./data/external"
DOWNLOAD_ALL=false
DRY_RUN=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --all)
            DOWNLOAD_ALL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            head -n 17 "$0" | tail -n +2 | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Colours (disabled when stdout is not a terminal)
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then
    BOLD="\033[1m"
    GREEN="\033[0;32m"
    YELLOW="\033[0;33m"
    RED="\033[0;31m"
    CYAN="\033[0;36m"
    RESET="\033[0m"
else
    BOLD="" GREEN="" YELLOW="" RED="" CYAN="" RESET=""
fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo -e "${BOLD}${CYAN}"
echo "========================================================================"
echo "  BL1 — Maxwell HD-MEA Dataset Downloader"
echo "========================================================================"
echo -e "${RESET}"
echo "Target directory : ${DATA_DIR}"
echo "Download all     : ${DOWNLOAD_ALL}"
echo "Dry run          : ${DRY_RUN}"
echo ""
echo "Datasets (default):"
echo "  1. Rat cortical cultures  (DANDI 001611)   ~39.9 GB"
echo "  2. DishBrain spike data   (OSF 5u6qv)      size varies"
echo "  3. Sharf 2022 organoid    (Zenodo 6578989) ~70.9 GB"
if $DOWNLOAD_ALL; then
    echo ""
    echo "Datasets (--all):"
    echo "  4. Protosequences         (DANDI 001603)  ~322.2 GB"
    echo "  5. SpikeCanvas            (DANDI 001747)  ~1.6 TB"
fi
echo ""
echo "------------------------------------------------------------------------"
echo ""

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
info()  { echo -e "${GREEN}[INFO]${RESET}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error() { echo -e "${RED}[ERROR]${RESET} $*"; }

# Track results for the summary
declare -a RESULTS=()

record_result() {
    # $1 = dataset name, $2 = status (OK | SKIPPED | FAILED | DRY-RUN)
    RESULTS+=("$1|$2")
}

# ---------------------------------------------------------------------------
# Check / install prerequisites
# ---------------------------------------------------------------------------
check_prerequisites() {
    info "Checking prerequisites ..."

    # wget or curl — at least one is needed
    if command -v wget &>/dev/null; then
        DOWNLOADER="wget"
    elif command -v curl &>/dev/null; then
        DOWNLOADER="curl"
    else
        error "Neither wget nor curl found. Please install one of them."
        exit 1
    fi
    info "HTTP downloader: ${DOWNLOADER}"

    # dandi CLI
    if ! command -v dandi &>/dev/null; then
        warn "dandi CLI not found."
        if $DRY_RUN; then
            warn "(dry-run) Would install dandi via: pip install dandi"
        else
            info "Installing dandi via pip ..."
            pip install dandi || {
                error "Failed to install dandi. Install manually: pip install dandi"
                exit 1
            }
        fi
    fi
    if command -v dandi &>/dev/null; then
        info "dandi CLI: $(dandi --version 2>&1 | head -1)"
    fi

    echo ""
}

# ---------------------------------------------------------------------------
# 1. Rat cortical cultures — DANDI 001611 (~39.9 GB)
# ---------------------------------------------------------------------------
download_dandi_001611() {
    local name="Rat cortical cultures (DANDI 001611)"
    local dest="${DATA_DIR}/dandi_001611_rat_cortical"
    local size="~39.9 GB"

    echo -e "${BOLD}>>> ${name}  [${size}]${RESET}"

    if $DRY_RUN; then
        info "(dry-run) dandi download https://dandiarchive.org/dandiset/001611 -o ${dest}"
        record_result "${name}" "DRY-RUN"
        echo ""
        return 0
    fi

    mkdir -p "${dest}"
    if dandi download https://dandiarchive.org/dandiset/001611 -o "${dest}"; then
        info "${name} — download complete."
        record_result "${name}" "OK"
    else
        error "${name} — download failed (will continue with next dataset)."
        record_result "${name}" "FAILED"
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# 2. DishBrain spike data — OSF 5u6qv
# ---------------------------------------------------------------------------
download_osf_dishbrain() {
    local name="DishBrain spike data (OSF 5u6qv)"
    local dest="${DATA_DIR}/osf_dishbrain"
    local size="size varies"

    echo -e "${BOLD}>>> ${name}  [${size}]${RESET}"

    if $DRY_RUN; then
        info "(dry-run) Would download from https://osf.io/5u6qv/ into ${dest}"
        record_result "${name}" "DRY-RUN"
        echo ""
        return 0
    fi

    mkdir -p "${dest}"

    # Try wget recursive download first
    if [[ "${DOWNLOADER}" == "wget" ]]; then
        info "Attempting recursive wget download from OSF ..."
        if wget -r -np -nH --cut-dirs=1 \
                -P "${dest}" \
                "https://files.osf.io/v1/resources/5u6qv/providers/osfstorage/" 2>&1; then
            info "${name} — download complete."
            record_result "${name}" "OK"
            echo ""
            return 0
        else
            warn "wget recursive download failed. Falling back to manual instructions."
        fi
    fi

    # Fallback: print manual instructions
    warn "Automatic download of OSF data was not successful."
    echo ""
    echo "  To download manually:"
    echo "    1. Visit https://osf.io/5u6qv/"
    echo "    2. Click 'Download as zip' or download individual files."
    echo "    3. Place the files in: ${dest}"
    echo ""
    record_result "${name}" "SKIPPED (manual download needed)"
    echo ""
}

# ---------------------------------------------------------------------------
# 3. Sharf 2022 organoid — Zenodo 6578989 (~70.9 GB)
# ---------------------------------------------------------------------------
download_zenodo_sharf() {
    local name="Sharf 2022 organoid (Zenodo 6578989)"
    local dest="${DATA_DIR}/zenodo_sharf_2022"
    local size="~70.9 GB"

    echo -e "${BOLD}>>> ${name}  [${size}]${RESET}"

    if $DRY_RUN; then
        info "(dry-run) Would download Zenodo record 6578989 into ${dest}"
        record_result "${name}" "DRY-RUN"
        echo ""
        return 0
    fi

    mkdir -p "${dest}"

    # Try zenodo_get first (pip install zenodo_get) — it handles multi-file records
    if command -v zenodo_get &>/dev/null; then
        info "Using zenodo_get to download record 6578989 ..."
        if zenodo_get -d 6578989 -o "${dest}"; then
            info "${name} — download complete."
            record_result "${name}" "OK"
            echo ""
            return 0
        else
            warn "zenodo_get failed; falling back to direct download."
        fi
    fi

    # Fallback: fetch the file listing from the Zenodo API and download each file
    info "Fetching file list from Zenodo API ..."
    local api_url="https://zenodo.org/api/records/6578989"
    local file_list

    if [[ "${DOWNLOADER}" == "curl" ]]; then
        file_list=$(curl -sL "${api_url}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for f in data.get('files', []):
    print(f['links']['self'] + '|' + f['key'] + '|' + str(f.get('size', 0)))
" 2>/dev/null) || true
    else
        file_list=$(wget -qO- "${api_url}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for f in data.get('files', []):
    print(f['links']['self'] + '|' + f['key'] + '|' + str(f.get('size', 0)))
" 2>/dev/null) || true
    fi

    if [[ -z "${file_list}" ]]; then
        warn "Could not retrieve file list from Zenodo API."
        echo ""
        echo "  To download manually:"
        echo "    1. Visit https://zenodo.org/records/6578989"
        echo "    2. Download all files into: ${dest}"
        echo ""
        record_result "${name}" "SKIPPED (manual download needed)"
        echo ""
        return 0
    fi

    local download_ok=true
    while IFS='|' read -r url filename filesize; do
        local size_human
        size_human=$(python3 -c "
s=${filesize}
for u in ['B','KB','MB','GB','TB']:
    if s < 1024: print(f'{s:.1f} {u}'); break
    s /= 1024
" 2>/dev/null || echo "${filesize} B")

        info "Downloading ${filename} (${size_human}) ..."
        if [[ "${DOWNLOADER}" == "wget" ]]; then
            wget -c -q --show-progress -O "${dest}/${filename}" "${url}" || {
                warn "Failed to download ${filename}"
                download_ok=false
            }
        else
            curl -L -C - -o "${dest}/${filename}" "${url}" || {
                warn "Failed to download ${filename}"
                download_ok=false
            }
        fi
    done <<< "${file_list}"

    if $download_ok; then
        info "${name} — download complete."
        record_result "${name}" "OK"
    else
        error "${name} — some files failed to download."
        record_result "${name}" "FAILED (partial)"
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# 4. Protosequences — DANDI 001603 (~322.2 GB)  [--all only]
# ---------------------------------------------------------------------------
download_dandi_001603() {
    local name="Protosequences (DANDI 001603)"
    local dest="${DATA_DIR}/dandi_001603_protosequences"
    local size="~322.2 GB"

    echo -e "${BOLD}>>> ${name}  [${size}]${RESET}"

    if $DRY_RUN; then
        info "(dry-run) dandi download https://dandiarchive.org/dandiset/001603 -o ${dest}"
        record_result "${name}" "DRY-RUN"
        echo ""
        return 0
    fi

    mkdir -p "${dest}"
    if dandi download https://dandiarchive.org/dandiset/001603 -o "${dest}"; then
        info "${name} — download complete."
        record_result "${name}" "OK"
    else
        error "${name} — download failed (will continue with next dataset)."
        record_result "${name}" "FAILED"
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# 5. SpikeCanvas — DANDI 001747 (~1.6 TB)  [--all only]
# ---------------------------------------------------------------------------
download_dandi_001747() {
    local name="SpikeCanvas (DANDI 001747)"
    local dest="${DATA_DIR}/dandi_001747_spikecanvas"
    local size="~1.6 TB"

    echo -e "${BOLD}>>> ${name}  [${size}]${RESET}"

    if $DRY_RUN; then
        info "(dry-run) dandi download https://dandiarchive.org/dandiset/001747 -o ${dest}"
        record_result "${name}" "DRY-RUN"
        echo ""
        return 0
    fi

    mkdir -p "${dest}"
    if dandi download https://dandiarchive.org/dandiset/001747 -o "${dest}"; then
        info "${name} — download complete."
        record_result "${name}" "OK"
    else
        error "${name} — download failed."
        record_result "${name}" "FAILED"
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Update .gitignore
# ---------------------------------------------------------------------------
update_gitignore() {
    local gitignore
    # Walk up to find the repo root (where .git/ lives)
    gitignore="$(git rev-parse --show-toplevel 2>/dev/null)/.gitignore" || gitignore=".gitignore"

    if [[ -f "${gitignore}" ]]; then
        if ! grep -qxF 'data/' "${gitignore}"; then
            if $DRY_RUN; then
                info "(dry-run) Would add 'data/' to ${gitignore}"
            else
                echo 'data/' >> "${gitignore}"
                info "Added 'data/' to ${gitignore}"
            fi
        else
            info "'data/' already in ${gitignore}"
        fi
    else
        if $DRY_RUN; then
            info "(dry-run) Would create ${gitignore} with 'data/'"
        else
            echo 'data/' > "${gitignore}"
            info "Created ${gitignore} with 'data/'"
        fi
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print_summary() {
    echo -e "${BOLD}${CYAN}"
    echo "========================================================================"
    echo "  Download Summary"
    echo "========================================================================"
    echo -e "${RESET}"

    printf "  %-50s %s\n" "DATASET" "STATUS"
    printf "  %-50s %s\n" "-------" "------"
    for entry in "${RESULTS[@]}"; do
        local dname="${entry%%|*}"
        local status="${entry##*|}"
        case "${status}" in
            OK)       printf "  %-50s ${GREEN}%s${RESET}\n" "${dname}" "${status}" ;;
            FAILED*)  printf "  %-50s ${RED}%s${RESET}\n"   "${dname}" "${status}" ;;
            DRY-RUN)  printf "  %-50s ${YELLOW}%s${RESET}\n" "${dname}" "${status}" ;;
            *)        printf "  %-50s ${YELLOW}%s${RESET}\n" "${dname}" "${status}" ;;
        esac
    done

    echo ""
    if ! $DRY_RUN; then
        if [[ -d "${DATA_DIR}" ]]; then
            local total_size
            total_size=$(du -sh "${DATA_DIR}" 2>/dev/null | cut -f1 || echo "unknown")
            info "Total size of ${DATA_DIR}: ${total_size}"
        fi
    fi

    echo ""
    echo "Notes:"
    echo "  - dandi download supports resuming by default. Re-run to continue"
    echo "    any interrupted downloads."
    echo "  - wget/curl downloads use -c / -C - for resume support."
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    check_prerequisites
    update_gitignore

    # Default datasets (always downloaded)
    download_dandi_001611
    download_osf_dishbrain
    download_zenodo_sharf

    # Optional large datasets (--all only)
    if $DOWNLOAD_ALL; then
        download_dandi_001603
        download_dandi_001747
    fi

    print_summary
}

main
