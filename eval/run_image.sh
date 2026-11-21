#!/usr/bin/env bash
# Runs the image once under the scorer's limits on a sample directory and scores it.
#   eval/run_image.sh <image> <sample dir> <output.jsonl> [score.py]
set -euo pipefail
image=$1 sample=$(cd "$2" && pwd) output=$3 scorer=${4:-}
docker run --rm --platform linux/amd64 --network none --cpus 2 \
  --memory 4096m --memory-swap 4096m --pids-limit 256 --read-only \
  --tmpfs /tmp:rw,size=64m,mode=1777 --cap-drop ALL \
  --security-opt no-new-privileges --user 65534:65534 --env HOME=/tmp \
  --volume "$sample/commits:/data/commits:ro" \
  --volume "$sample/tests.json:/data/tests.json:ro" \
  "$image" python -S -B -c '
import resource, sys, time
sys.path.insert(0, "/app")
start = time.time()
import blast_select
blast_select.main()
r = resource.getrusage(resource.RUSAGE_SELF)
c = resource.getrusage(resource.RUSAGE_CHILDREN)
print("cpu %.3f s (+%.3f s in subprocesses)  wall %.3f s  peak %d MB" % (r.ru_utime + r.ru_stime, c.ru_utime + c.ru_stime, time.time() - start, r.ru_maxrss // 1024), file=sys.stderr)
' > "$output"
if [ -n "$scorer" ]; then python3 "$scorer" "$sample" "$sample/answers.json" "$output"; fi
