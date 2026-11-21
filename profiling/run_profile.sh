#!/usr/bin/env bash
# Runs the sqlglot suite under the tracing plugins.
#   profiling/run_profile.sh <profile image> <sqlglot checkout> <output dir>
# test_max_nodes needs about 1.7 GB on its own, so it runs in a second container.
set -euo pipefail
image=$1 src=$(cd "$2" && pwd) out=$(mkdir -p "$3" && cd "$3" && pwd)
plugins=$(cd "$(dirname "$0")" && pwd)
chmod 777 "$out"
run() {
  local name=$1 memory=$2; shift 2
  docker run --rm --cpus 2 --memory "$memory" --memory-swap "$memory" \
    -v "$src:/src:ro" -v "$plugins:/plugins:ro" -v "$out:/out" \
    -e PYTHONPATH=/src:/plugins -e PROFILE_OUT="/out/$name.txt.gz" -e OUTCOME_OUT="/out/$name.json" \
    -w /src "$image" python -m pytest -q -p trace_plugin -p outcome_plugin -p no:cacheprovider \
    --timeout 900 "$@" > "$out/$name.log" 2>&1
  tail -1 "$out/$name.log"
}
run main 2800m tests --deselect tests/test_parser.py::TestParser::test_max_nodes
run max_nodes 3500m tests/test_parser.py::TestParser::test_max_nodes
