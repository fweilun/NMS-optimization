#!/bin/bash
EXEC=$1
shift

echo "===== Warm Up ====="
./"$EXEC" "$@"

echo
echo "===== Profiling ====="
perf stat -e cache-references,cache-misses,LLC-loads,LLC-load-misses ./"$EXEC" "$@"