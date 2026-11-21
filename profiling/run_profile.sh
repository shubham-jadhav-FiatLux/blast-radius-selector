#!/usr/bin/env bash
# Runs the sqlglot suite under the tracing plugin, twice:
#   trace  line coverage, file reads, validation, class attribute reads, dict keys and the
#          classes each test creates or generates SQL for
#   weak   comparisons and boolean operators evaluated against their swapped forms
#   profiling/run_profile.sh <profile image> <sqlglot checkout> <output dir> [seed]
# test_max_nodes needs about 1.7 GB on its own, so each pass runs it in a second container.
set -euo pipefail
image=$1 src=$(cd "$2" && pwd) out=$(mkdir -p "$3" && cd "$3" && pwd) seed=${4:-0}
plugins=$(cd "$(dirname "$0")" && pwd)
chmod 777 "$out"
run() {
  local name=$1 mode=$2 memory=$3; shift 3
  local keys=0; [ "$mode" = PROFILE_READS ] && keys=1  # the trace pass also records dict keys and class users
  docker run --rm --cpus 2 --memory "$memory" --memory-swap "$memory" \
    -v "$src:/src:ro" -v "$plugins:/plugins:ro" -v "$out:/out" \
    -e PYTHONPATH=/src:/plugins -e "$mode=1" -e PROFILE_KEYS="$keys" -e PROFILE_USERS="$keys" -e PYTHONHASHSEED="$seed" \
    -e PROFILE_OUT="/out/$name.txt.gz" -e OUTCOME_OUT="/out/$name.json" \
    -w /src "$image" python -m pytest -q -p trace_plugin -p outcome_plugin -p no:cacheprovider \
    --timeout 1800 "$@" > "$out/$name.log" 2>&1
  tail -1 "$out/$name.log"
}
max_nodes=tests/test_parser.py::TestParser::test_max_nodes
run trace_main PROFILE_READS 2800m tests --deselect "$max_nodes"
run trace_max_nodes PROFILE_READS 3500m "$max_nodes"
run weak_main PROFILE_WEAK 2800m tests --deselect "$max_nodes"
run weak_max_nodes PROFILE_WEAK 3500m "$max_nodes"
