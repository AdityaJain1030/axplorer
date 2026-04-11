#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_kfour_production.sh
# KFour production sweep — A40 (48 GB VRAM, 16 CPU cores)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Phases
#   1. VALIDATION  — short runs against 5 known-optimal complement edge counts
#   2. PRODUCTION  — full sweep over all feasible (N, t) pairs
#   3. SUMMARY     — compute c_lb from saved scores, write results/kfour/summary.csv
#
# Resumability
#   • train.py already auto-resumes via epoch.txt inside the dump directory.
#   • A run is considered complete when best_score.txt exists in its output
#     directory.  Complete runs are silently skipped on re-invocation.
#
# Usage
#   ./run_kfour_production.sh                  # run everything
#   ./run_kfour_production.sh --val-only       # validation phase only
#   ./run_kfour_production.sh --prod-only      # production + summary
#   ./run_kfour_production.sh --summary-only   # recompute summary from saved scores
#
# Assumptions
#   • Script lives in the axplorer repo root (same directory as train.py)
#   • micromamba environment "env_axplorer" is activated / importable
#   • CUDA device visible to PyTorch  (--cpu flag omitted intentionally)
#
# Score extraction
#   environment.py:compute_stats logs "Max score: <N>" once per epoch via the
#   Python logger.  We grep the tee'd training log for this pattern and take
#   the maximum across all epochs.
# ═══════════════════════════════════════════════════════════════════════════════

# -e omitted deliberately: individual train.py runs may fail; the sweep should
# continue.  -u catches typos in variable names; -o pipefail catches pipe errors.
set -uo pipefail

# ─── Paths ─────────────────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="${REPO}/results/kfour"
VAL_DIR="${RESULTS_ROOT}/validation"
PROD_DIR="${RESULTS_ROOT}/production"
GLOBAL_LOG="${RESULTS_ROOT}/run.log"

mkdir -p "${VAL_DIR}" "${PROD_DIR}"

# ─── Server tuning ─────────────────────────────────────────────────────────────
CONDA_ENV="env_axplorer"
NUM_CPUS=16        # data-generation worker pool (matches --num_workers)

# Resolve Python runner.  Prefer micromamba; fall back to conda or bare python.
if command -v micromamba &>/dev/null; then
    PY() { micromamba run -n "${CONDA_ENV}" python "$@"; }
elif command -v conda &>/dev/null; then
    PY() { conda run --no-capture-output -n "${CONDA_ENV}" python "$@"; }
else
    # Assume the correct virtualenv is already active
    PY() { python "$@"; }
fi

# ─── Argument parsing ──────────────────────────────────────────────────────────
DO_VAL=true
DO_PROD=true
DO_SUMMARY=true

for arg in "$@"; do
    case "${arg}" in
        --val-only)     DO_PROD=false;    DO_SUMMARY=false ;;
        --prod-only)    DO_VAL=false ;;
        --summary-only) DO_VAL=false;    DO_PROD=false ;;
        *) echo "Unknown option: ${arg}"; exit 1 ;;
    esac
done

# ─── Logging ───────────────────────────────────────────────────────────────────
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}"
    echo "${msg}" >> "${GLOBAL_LOG}"
}

# ─── Score extraction ──────────────────────────────────────────────────────────
# environment.py:compute_stats emits exactly one line per epoch:
#   "Max score: <integer>"
# We collect all such lines from the training log and return the maximum.
# Falls back to -1 if the file is absent or no match is found.
extract_best_score() {
    local log_file="$1"
    if [[ ! -f "${log_file}" ]]; then
        echo "-1"; return
    fi
    # Pattern matches both "Max score: 68" and "Max score: 68.0"
    local best
    best=$(grep -oP '(?<=Max score: )\S+' "${log_file}" 2>/dev/null \
           | awk 'BEGIN{m=-1} {v=int($1); if(v>m) m=v} END{print m}')
    echo "${best:--1}"
}

# ─── Core training wrapper ─────────────────────────────────────────────────────
# run_training <out_dir> <N> <t> [extra train.py flags...]
#
# Writes:
#   <out_dir>/train.log        — tee'd stdout+stderr from train.py
#   <out_dir>/best_score.txt   — single integer, the best score seen
#   <out_dir>/checkpoint/      — model checkpoints (train.py resumes from here)
#
# Returns 0 even on non-zero exit from train.py (logged as WARNING).
run_training() {
    local out_dir="$1" N="$2" t="$3"
    shift 3

    mkdir -p "${out_dir}"
    local log_file="${out_dir}/train.log"
    local dump_path="${out_dir}/checkpoint"
    local exp_name="kfour_N${N}_t${t}"

    log "  START  N=${N}  t=${t}  out=${out_dir}"

    # train.py auto-resumes if dump_path/epoch.txt already exists
    PY "${REPO}/train.py" \
        --env_name kfour \
        --N "${N}" \
        --t "${t}" \
        --exp_name "${exp_name}" \
        --dump_path "${dump_path}" \
        "$@" 2>&1 | tee -a "${log_file}" \
    || log "  WARNING  train.py exited non-zero for N=${N} t=${t} — partial results saved"

    local best
    best=$(extract_best_score "${log_file}")
    echo "${best}" > "${out_dir}/best_score.txt"
    log "  DONE   N=${N}  t=${t}  best_score=${best}"
}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. VALIDATION PHASE
# ═══════════════════════════════════════════════════════════════════════════════
# Five known-optimal (N, t) instances with published complement edge counts.
# We run moderate training and check whether the model matches or exceeds them.
#
#   N=17, t=4 → 68 edges   (c_lb ≈ 0.679)
#   N=18, t=5 → 99 edges   (c_lb ≈ 0.744)
#   N=20, t=5 → 120 edges  (c_lb ≈ 0.719)
#   N=22, t=5 → 132 edges  (c_lb ≈ 0.745)
#   N=24, t=7 → 228 edges  (c_lb ≈ 0.721)
#
# If best_score > known target → NEW BOUND (print a loud flag).
# Validation results land in results/kfour/validation/<N>_<t>/ and do NOT
# block production runs for the same (N, t) pair.
#
# Parameter rationale
#   max_epochs=150 : enough to converge on ≤24-vertex graphs
#   max_steps=5000 : ~33 steps/epoch cap keeps each run under ~10 min on A40
#   n_layer=4 n_embd=256 : lightweight model sufficient for validation
#   max_len=300    : Turán T(24,6)=240 edges < 300 for all validation targets
# ═══════════════════════════════════════════════════════════════════════════════

if ${DO_VAL}; then
    log "════════════════════════════════════════════════════════"
    log "PHASE 1: VALIDATION"
    log "════════════════════════════════════════════════════════"

    # Moderate parameters — enough to reach known targets, not production scale
    VAL_COMMON=(
        --max_epochs    150
        --max_steps     5000
        --gensize       8000
        --num_samples_from_model 3000
        --pop_size      8000
        --batch_size    64
        --n_layer       4
        --n_embd        256
        --max_len       300
        --num_workers   "${NUM_CPUS}"
        --process_pool  true
        --always_search true
        --encoding_tokens single_integer
    )

    # Format: "<N>_<t> <known_target>"
    declare -A VAL_TARGETS=(
        ["17_4"]=68
        ["18_5"]=99
        ["20_5"]=120
        ["22_5"]=132
        ["24_7"]=228
    )

    for key in "${!VAL_TARGETS[@]}"; do
        N="${key%%_*}"
        t="${key##*_}"
        target="${VAL_TARGETS[$key]}"
        out_dir="${VAL_DIR}/${key}"

        log "── Validation  N=${N}  t=${t}  known_target=${target} ──"
        run_training "${out_dir}" "${N}" "${t}" "${VAL_COMMON[@]}"

        best=$(cat "${out_dir}/best_score.txt")

        if [[ "${best}" -gt "${target}" ]] 2>/dev/null; then
            log "  ★★★ NEW BOUND  N=${N}  t=${t}  best=${best}  >  known=${target}  ★★★"
        elif [[ "${best}" -eq "${target}" ]] 2>/dev/null; then
            log "  ✓   MATCHED    N=${N}  t=${t}  best=${best}  == known=${target}"
        else
            log "  ✗   BELOW      N=${N}  t=${t}  best=${best}  <  known=${target}  (may need more epochs)"
        fi
    done

    log "PHASE 1 COMPLETE"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PRODUCTION SWEEP
# ═══════════════════════════════════════════════════════════════════════════════
# Full-scale training over all feasible (N, t) pairs.
#
# Feasibility bounds (N strictly less than R(4, t)):
#   t=5  R(4,5)=25   → N ∈ {18 … 24}
#   t=6  R(4,6)=36   → N ∈ {25 … 35}
#   t=7  R(4,7)=49   → N ∈ {36 … 48}
#   t=8  R(4,8)>55   → N ∈ {49 … 55}   (conservative upper bound)
#
# Model / batch sizing for A40 (48 GB VRAM)
# ─────────────────────────────────────────
# Sequence length = |E(H)| + 3 special tokens.  The densest K_t-free graph
# is the Turán graph T(N, t-1), with (t-2)/(t-1) · N²/2 edges:
#
#   Tier   t   N_max   Turán_max_edges   max_len   n_embd  n_layer  batch
#   ──────────────────────────────────────────────────────────────────────
#    A     5    24       216              300        512      8       64
#    B     6    35       490              600        512      8       32
#    C     7    48       960             1100        384      6       16
#    D     8    55      1296             1500        256      6        8
#
# Memory estimate (worst case per tier, float32 activations):
#   A: 64 × 303 × 512 × 8L ≈ 3.1 GB     (well within 48 GB)
#   B: 32 × 603 × 512 × 8L ≈ 3.1 GB
#   C: 16 × 1103 × 384 × 6L ≈ 1.5 GB
#   D:  8 × 1503 × 256 × 6L ≈ 0.5 GB
# (Actual usage is higher once optimizer states are included, but still safe.)
#
# gensize / pop_size: larger values improve the diversity of training data.
# num_samples_from_model: how many sequences the model generates each epoch.
# ═══════════════════════════════════════════════════════════════════════════════

# Returns space-separated extra args for a given N (tier A–D).
prod_model_args() {
    local N=$1
    if   (( N <= 24 )); then
        # Tier A: t=5, small graphs, largest model
        echo "--n_layer 8  --n_embd 512  --batch_size 64  --max_len 300"
        echo "--gensize 20000  --pop_size 20000  --num_samples_from_model 6000"
    elif (( N <= 35 )); then
        # Tier B: t=6, medium graphs
        echo "--n_layer 8  --n_embd 512  --batch_size 32  --max_len 600"
        echo "--gensize 12000  --pop_size 12000  --num_samples_from_model 4000"
    elif (( N <= 48 )); then
        # Tier C: t=7, large graphs — reduce model width to stay in VRAM
        echo "--n_layer 6  --n_embd 384  --batch_size 16  --max_len 1100"
        echo "--gensize 6000   --pop_size 6000   --num_samples_from_model 2000"
    else
        # Tier D: t=8, very large graphs — further reduced model
        echo "--n_layer 6  --n_embd 256  --batch_size 8   --max_len 1500"
        echo "--gensize 3000   --pop_size 3000   --num_samples_from_model 1000"
    fi
}

# Runs one (N, t) production pair; skips if best_score.txt already exists.
sweep_one() {
    local N=$1 t=$2
    local out_dir="${PROD_DIR}/${N}_${t}"
    local score_file="${out_dir}/best_score.txt"

    if [[ -f "${score_file}" ]]; then
        local existing; existing=$(cat "${score_file}")
        log "SKIP  N=${N}  t=${t}  (best_score=${existing} already recorded)"
        return 0
    fi

    # Build model-size args array from the tier helper
    local model_args_str
    model_args_str=$(prod_model_args "${N}")
    # shellcheck disable=SC2206  # intentional word-splitting of flag pairs
    read -ra model_args <<< "${model_args_str}"

    run_training "${out_dir}" "${N}" "${t}" \
        --max_epochs    500 \
        --max_steps     100000 \
        --num_workers   "${NUM_CPUS}" \
        --process_pool  true \
        --always_search true \
        --encoding_tokens single_integer \
        "${model_args[@]}"
}

if ${DO_PROD}; then
    log "════════════════════════════════════════════════════════"
    log "PHASE 2: PRODUCTION SWEEP"
    log "════════════════════════════════════════════════════════"

    # ── t=5  (R(4,5) = 25, N < 25) ────────────────────────────────────────────
    log "── t=5  N=18..24 ──"
    for N in $(seq 18 24); do sweep_one "${N}" 5; done

    # ── t=6  (R(4,6) = 36, N < 36) ────────────────────────────────────────────
    log "── t=6  N=25..35 ──"
    for N in $(seq 25 35); do sweep_one "${N}" 6; done

    # ── t=7  (R(4,7) = 49, N < 49) ────────────────────────────────────────────
    log "── t=7  N=36..48 ──"
    for N in $(seq 36 48); do sweep_one "${N}" 7; done

    # ── t=8  (R(4,8) > 55, conservative upper bound) ──────────────────────────
    log "── t=8  N=49..55 ──"
    for N in $(seq 49 55); do sweep_one "${N}" 8; done

    log "PHASE 2 COMPLETE"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# 3. POST-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
# Reconstruct graph statistics from complement edge counts and compute the
# conjecture lower bound constant for each completed production run.
#
# Formulas
#   H is the complement of G (what the model builds).
#   E_G   = N*(N-1)/2 - |E(H)|     edges in the original graph G
#   d_avg = 2 * E_G / N             average degree in G
#   c_lb  = (t-1) * d_avg / (N * ln(d_avg))
#               lower bound on the conjecture constant c
#               (higher c_lb is a tighter bound, target: prove c ≥ some constant)
#
# For each N, c_min(N) = min over feasible t of c_lb(N, t) is the tightest
# bound achievable for that vertex count.
#
# Output: printed table + results/kfour/summary.csv
# ═══════════════════════════════════════════════════════════════════════════════

if ${DO_SUMMARY}; then
    log "════════════════════════════════════════════════════════"
    log "PHASE 3: POST-PROCESSING"
    log "════════════════════════════════════════════════════════"

    # Pass directories via env vars so the heredoc does not need shell expansion
    export KFOUR_PROD_DIR="${PROD_DIR}"
    export KFOUR_SUMMARY_CSV="${RESULTS_ROOT}/summary.csv"
    export KFOUR_GLOBAL_LOG="${GLOBAL_LOG}"

    PY - <<'PYEOF'
import os, math, csv, datetime
from collections import defaultdict

prod_dir   = os.environ["KFOUR_PROD_DIR"]
csv_out    = os.environ["KFOUR_SUMMARY_CSV"]
global_log = os.environ["KFOUR_GLOBAL_LOG"]

rows   = []
c_by_N = defaultdict(list)

# ── Collect results ──────────────────────────────────────────────────────────
for entry in sorted(os.listdir(prod_dir)):
    score_file = os.path.join(prod_dir, entry, "best_score.txt")
    if not os.path.isfile(score_file):
        continue

    try:
        N_str, t_str = entry.split("_", 1)
        N, t = int(N_str), int(t_str)
    except ValueError:
        print(f"  SKIP  unrecognised directory: {entry}")
        continue

    raw = open(score_file).read().strip()
    try:
        best_H = int(raw)
    except ValueError:
        print(f"  SKIP  unparseable score '{raw}' in {entry}")
        continue

    if best_H <= 0:
        print(f"  SKIP  N={N} t={t}  score={best_H}  (no valid graph found)")
        continue

    # --- Graph statistics ---
    # Total possible edges in K_N
    K_N_edges = N * (N - 1) // 2
    # Edges in original graph G = complement of H
    E_G = K_N_edges - best_H
    if E_G <= 0:
        print(f"  SKIP  N={N} t={t}  E_G={E_G}  (H is denser than K_N?)")
        continue

    d_avg = 2.0 * E_G / N
    if d_avg <= 1.0:
        # c_lb formula requires d_avg > 1 (log undefined / negative for d_avg ≤ 1)
        print(f"  SKIP  N={N} t={t}  d_avg={d_avg:.3f} ≤ 1  (graph too sparse for c_lb)")
        continue

    # Lower bound on the conjecture constant c (maximise over all valid H)
    c_lb = (t - 1) * d_avg / (N * math.log(d_avg))

    row = {
        "N":            N,
        "t":            t,
        "best_score_H": best_H,
        "E_G":          E_G,
        "d_avg":        round(d_avg, 4),
        "c_lb":         round(c_lb, 6),
    }
    rows.append(row)
    c_by_N[N].append((t, c_lb))

rows.sort(key=lambda r: (r["N"], r["t"]))

# ── Per-(N, t) table ─────────────────────────────────────────────────────────
SEP = "─" * 66
HDR = f"{'N':>4}  {'t':>3}  {'|E(H)|':>8}  {'E(G)':>7}  {'d_avg':>8}  {'c_lb':>10}"

print()
print("=" * 66)
print("  Per-(N, t) production results")
print("=" * 66)
print(HDR)
print(SEP)
for r in rows:
    print(f"{r['N']:>4}  {r['t']:>3}  {r['best_score_H']:>8}  "
          f"{r['E_G']:>7}  {r['d_avg']:>8.4f}  {r['c_lb']:>10.6f}")
print(SEP)

# ── Per-N minimum c_lb ───────────────────────────────────────────────────────
print()
print("=" * 44)
print("  c_min(N) = min_t  c_lb(N, t)")
print("=" * 44)
print(f"{'N':>4}  {'best_t':>6}  {'c_min':>10}")
print("─" * 44)
global_min_c = None
global_min_entry = None
for N in sorted(c_by_N):
    best_t, c_min = min(c_by_N[N], key=lambda x: x[1])
    print(f"{N:>4}  {best_t:>6}  {c_min:>10.6f}")
    if global_min_c is None or c_min < global_min_c:
        global_min_c = c_min
        global_min_entry = (N, best_t, c_min)
print("─" * 44)

if global_min_entry is not None:
    gN, gt, gc = global_min_entry
    print(f"  Global c_lb minimum: {gc:.6f}  (N={gN}, t={gt})")

# ── Write CSV ────────────────────────────────────────────────────────────────
fieldnames = ["N", "t", "best_score_H", "E_G", "d_avg", "c_lb"]
with open(csv_out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

summary_msg = (
    f"Summary written: {csv_out}  "
    f"({len(rows)} entries, "
    f"global_c_lb_min={global_min_c:.6f} at N={global_min_entry[0]} t={global_min_entry[1]})"
    if global_min_entry else f"Summary written: {csv_out}  ({len(rows)} entries)"
)
print()
print(summary_msg)

# Also append the summary line to the global run log
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(global_log, "a") as lf:
    lf.write(f"[{ts}] {summary_msg}\n")
PYEOF

    log "PHASE 3 COMPLETE"
fi

log "════════════════════════════════════════════════════════"
log "ALL PHASES COMPLETE"
log "Results:  ${RESULTS_ROOT}"
log "Summary:  ${RESULTS_ROOT}/summary.csv"
log "Run log:  ${GLOBAL_LOG}"
log "════════════════════════════════════════════════════════"
