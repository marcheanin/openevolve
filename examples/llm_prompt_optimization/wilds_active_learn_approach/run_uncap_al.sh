#!/usr/bin/env bash
# Active Learning experiment with uncapped train + 5000 candidate pool + val=450 + test=600
# Performs 8 AL cycles x 15 evolution iterations, then evaluates the best-by-val prompt
# on the FULL uncapped WILDS Amazon test split.
#
# Usage (run from this folder):
#   ./run_uncap_al.sh              # full run (8 cycles, ~6-9h on a single node)
#   ./run_uncap_al.sh smoke        # 2 cycles x 4 evolve iters; quick sanity check
#   ./run_uncap_al.sh resume       # continue an existing results dir
#   SKIP_FULL_TEST=1 ./run_uncap_al.sh   # skip full uncapped test at the end
#
# Override knobs via env (defaults shown):
#   N_AL=8 N_EVOLVE=15 RESULTS_DIR=results_all_categories_uncapped_train ./run_uncap_al.sh

set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
    echo "WARN: no .env file found; make sure OPENROUTER_API_KEY (or OPENAI_API_KEY) is set in env." >&2
fi

MODE="${1:-full}"
N_AL="${N_AL:-8}"
N_EVOLVE="${N_EVOLVE:-15}"
RESULTS_DIR="${RESULTS_DIR:-results_all_categories_uncapped_train}"
SKIP_FULL_TEST="${SKIP_FULL_TEST:-0}"

ARGS=(active_loop.py
      --config config_all_categories_uncapped_train.yaml
      --prompt initial_prompt_all_categories.txt
      --results-dir "$RESULTS_DIR"
      --no-evolve-early-stop)

case "$MODE" in
    smoke)
        ARGS+=(--smoke)
        ;;
    resume)
        ARGS+=(--n-al "$N_AL" --n-evolve "$N_EVOLVE" --resume-from-dir "$RESULTS_DIR")
        if [[ "$SKIP_FULL_TEST" != "1" ]]; then ARGS+=(--run-full-test); fi
        ;;
    full|*)
        ARGS+=(--n-al "$N_AL" --n-evolve "$N_EVOLVE")
        if [[ "$SKIP_FULL_TEST" != "1" ]]; then ARGS+=(--run-full-test); fi
        ;;
esac

echo "==> python ${ARGS[*]}"
START=$(date +%s)
python "${ARGS[@]}"
END=$(date +%s)
SECS=$((END - START))
printf 'Done in %02d:%02d:%02d\n' $((SECS/3600)) $(((SECS%3600)/60)) $((SECS%60))

cat <<EOF
Results: $RESULTS_DIR
  active_loop_log.json            - per-cycle metrics
  best_val_prompt.txt             - prompt with the best validation combined_score
  final_test_metrics.json         - capped (~600) test metrics for that prompt
  full_uncapped_test_metrics.json - FULL WILDS test metrics for the same prompt
  al_pool_manifest.json           - candidate pool composition (5000 reviews) for reproducibility
EOF
