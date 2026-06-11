#!/usr/bin/env bash
# Submit Scrib488 / Dpn555 / EdU647 fly-brain pipeline jobs on UVA HPC.
#
# Runs the same analysis as legacy analyze.sh jobruns:
#   DoG (sigma 1/12) -> MicroSAM (vit_l_lm) -> enrich -> coincidence (dice, 0.1, outline)
#
# Three modes:
#   pilot  - one .lif copied to scratch (test before full batch)
#   batch  - all Animal-*-scrib-dpn-edu.lif files under a Raw files folder
#   rerun  - resubmit only selected array task IDs from a prior batch file list
#
# Pilot example (copy to scratch first, recommended):
#   mkdir -p /scratch/zyh4up/batch-test/input
#   cp "/standard/vol191/siegristlab/Microsam_Segmentation/24h/AkhGal4 x OR Susie/Scrib488 Dpn555 EdU 647/Raw files/Animal-1-scrib-dpn-edu.lif" \
#      /scratch/zyh4up/batch-test/input/
#   export VISTIQ_SCRATCH=/scratch/zyh4up/batch-test
#   export VISTIQ_ENV=/scratch/zyh4up/vistiq-env-gpu
#   export SLURM_ACCOUNT=siegristlab
#   bash scripts/submit-scrib-pipeline.sh pilot \
#     /scratch/zyh4up/batch-test/input/Animal-1-scrib-dpn-edu.lif
#
# Full batch (all Animal-*-scrib-dpn-edu.lif in Raw files):
#   export SLURM_ACCOUNT=siegristlab
#   bash scripts/submit-scrib-pipeline.sh batch
#
# Rerun failed/cancelled array tasks only (after moving incomplete output aside):
#   sacct -j <JOBID> --format=JobID,State -n | grep -E 'FAILED|CANCELLED|TIMEOUT'
#   bash scripts/submit-scrib-pipeline.sh rerun 3,5,7
#   bash scripts/submit-scrib-pipeline.sh rerun-failed <JOBID>
#
# After git pull on the cluster, reinstall once:
#   pip install -e /path/to/vistiq

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE_SBATCH="$SCRIPT_DIR/pipeline.sbatch"
FILELIST_SH="$SCRIPT_DIR/filelist.sh"

VISTIQ_SCRATCH="${VISTIQ_SCRATCH:-/scratch/zyh4up/batch-test}"
VISTIQ_ENV="${VISTIQ_ENV:-/home/zyh4up/.conda/envs/vistiq-env-gpu}"
export LD_LIBRARY_PATH="$VISTIQ_ENV/lib:${LD_LIBRARY_PATH:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_TIME="${SLURM_TIME:-6:00:00}"
SCRIB_GLOB="${SCRIB_GLOB:-Animal-*-scrib-dpn-edu.lif}"

SCRIB_DATASET_ROOT="${SCRIB_DATASET_ROOT:-/standard/vol191/siegristlab/Microsam_Segmentation/24h/AkhGal4 x OR Susie/Scrib488 Dpn555 EdU 647}"
SCRIB_RAW_DIR="${SCRIB_RAW_DIR:-$SCRIB_DATASET_ROOT/Raw files}"
JOBRUN_ROOT="${JOBRUN_ROOT:-/standard/vol191/siegristlab/Microsam_Segmentation/jobruns/24h/AkhGal4 x OR Susie/Scrib488 Dpn555 EdU 647}"

ANIMAL1_LIF="$SCRIB_RAW_DIR/Animal-1-scrib-dpn-edu.lif"

usage() {
    cat <<EOF
Usage:
  $0 pilot <path/to/Animal-1-scrib-dpn-edu.lif> [output_subdir]
  $0 batch
  $0 rerun <array_spec> [filelist]
  $0 rerun-failed <slurm_job_id> [filelist]

  rerun examples:
    $0 rerun 3,5,7
    $0 rerun 3-5,8 /path/to/scrib-2026-06-08.filelist
    $0 rerun-failed 12345678

Environment variables:
  VISTIQ_SCRATCH   Scratch workspace (default: $VISTIQ_SCRATCH)
  VISTIQ_ENV       Conda env to activate in sbatch (default: $VISTIQ_ENV)
  SLURM_ACCOUNT    Slurm account, e.g. siegristlab (optional)
  SLURM_TIME       Job wall time (default: $SLURM_TIME)
  SCRIB_DATASET_ROOT  Dataset folder (default: AkhGal4 Scrib488 line)
  SCRIB_RAW_DIR       Raw files folder (default: \$SCRIB_DATASET_ROOT/Raw files)
  JOBRUN_ROOT         Output root (default: .../jobruns/24h/.../Scrib488 Dpn555 EdU 647)
  SCRIB_GLOB          LIF filename pattern (default: $SCRIB_GLOB)

Defaults for this dataset:
  Animal 1 LIF: $ANIMAL1_LIF
  Raw files:    $SCRIB_RAW_DIR
  Job output:   $JOBRUN_ROOT
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

mode="$1"
shift

submit_job() {
    local filelist="$1"
    local output_root="$2"
    local array_spec="$3"

    local count
    count="$(wc -l < "$filelist" | tr -d ' ')"
    if [[ "$count" -lt 1 ]]; then
        echo "Error: no files in $filelist"
        exit 1
    fi

    echo "Repository:   $REPO_ROOT"
    echo "File list:    $filelist ($count file(s))"
    echo "Output root:  $output_root"
    echo "Array:        $array_spec"
    echo "Environment:  $VISTIQ_ENV"
    echo ""
    echo "--- file list preview ---"
    head -n 5 "$filelist"
    if [[ "$count" -gt 5 ]]; then
        echo "... ($((count - 5)) more)"
    fi
    echo "-------------------------"

# Prefect runtime isolation is handled inside pipeline.sbatch, where SLURM_JOB_ID
# and SLURM_ARRAY_TASK_ID are available.

    local sbatch_args=(
        --export=ALL,VISTIQ_ENV="$VISTIQ_ENV"
        --array="$array_spec"
        --time="$SLURM_TIME"
    )
    if [[ -n "$SLURM_ACCOUNT" ]]; then
        sbatch_args=( -A "$SLURM_ACCOUNT" "${sbatch_args[@]}" )
    fi

    sbatch "${sbatch_args[@]}" "$PIPELINE_SBATCH" "$filelist" "$output_root"
}

resolve_batch_filelist() {
    if [[ $# -ge 1 && -n "${1:-}" ]]; then
        realpath "$1"
        return
    fi
    local filelist_dir
    filelist_dir="$(dirname "$JOBRUN_ROOT")/filelists"
    local latest
    latest="$(ls -t "$filelist_dir"/scrib-*.filelist 2>/dev/null | head -1 || true)"
    if [[ -z "$latest" ]]; then
        echo "Error: no scrib-*.filelist under $filelist_dir; pass filelist path" >&2
        exit 1
    fi
    echo "$latest"
}

failed_array_spec_from_job() {
    local job_id="$1"
    local spec
    spec="$(
        sacct -j "$job_id" --format=JobID,State -n -P 2>/dev/null | awk -F'|' '
            $2 ~ /^(FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)$/ {
                n = split($1, parts, "_")
                if (n >= 2 && parts[2] != "" && parts[2] !~ /batch/) print parts[2]
            }' | sort -n -u | paste -sd, -
    )"
    if [[ -z "$spec" ]]; then
        echo "Error: no failed/cancelled array tasks found for job $job_id" >&2
        exit 1
    fi
    echo "$spec"
}

case "$mode" in
    pilot)
        if [[ $# -lt 1 ]]; then
            echo "Error: pilot mode requires path to one .lif file"
            usage
            exit 1
        fi
        input_lif="$(realpath "$1")"
        if [[ ! -f "$input_lif" ]]; then
            echo "Error: input file not found: $input_lif"
            exit 1
        fi

        pilot_tag="${2:-pilot-animal1}"
        filelist_dir="$VISTIQ_SCRATCH/filelists"
        output_root="$VISTIQ_SCRATCH/output"
        mkdir -p "$filelist_dir" "$output_root"

        filelist="$filelist_dir/scrib-pilot.filelist"
        printf "'%s' '%s'\n" "$input_lif" "$pilot_tag" > "$filelist"

        echo "Pilot run: $input_lif"
        submit_job "$filelist" "$output_root" "1"
        ;;

    batch)
        if [[ ! -d "$SCRIB_RAW_DIR" ]]; then
            echo "Error: SCRIB_RAW_DIR not found: $SCRIB_RAW_DIR"
            exit 1
        fi

        filelist_dir="$(dirname "$JOBRUN_ROOT")/filelists"
        mkdir -p "$filelist_dir" "$JOBRUN_ROOT"

        timestamp="$(date +%Y-%m-%d)"
        filelist="$filelist_dir/scrib-${timestamp}.filelist"

        bash "$FILELIST_SH" "$SCRIB_RAW_DIR" "$filelist" "$SCRIB_GLOB"

        submit_job "$filelist" "$JOBRUN_ROOT" "1-$(wc -l < "$filelist" | tr -d ' ')"
        ;;

    rerun)
        if [[ $# -lt 1 ]]; then
            echo "Error: rerun mode requires array_spec (e.g. 3,5,7)"
            usage
            exit 1
        fi
        array_spec="$1"
        filelist="$(resolve_batch_filelist "${2:-}")"
        echo "Rerun array tasks: $array_spec"
        submit_job "$filelist" "$JOBRUN_ROOT" "$array_spec"
        ;;

    rerun-failed)
        if [[ $# -lt 1 ]]; then
            echo "Error: rerun-failed mode requires SLURM job ID"
            usage
            exit 1
        fi
        array_spec="$(failed_array_spec_from_job "$1")"
        filelist="$(resolve_batch_filelist "${2:-}")"
        echo "Rerun failed/cancelled tasks from job $1: $array_spec"
        submit_job "$filelist" "$JOBRUN_ROOT" "$array_spec"
        ;;

    *)
        echo "Error: unknown mode '$mode' (use pilot, batch, rerun, or rerun-failed)"
        usage
        exit 1
        ;;
esac
