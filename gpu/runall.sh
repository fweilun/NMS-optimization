#!/bin/bash
set -euo pipefail

all_files=(nms-opt-2000 nms-torch-solvebank64 nms-torch-solvebank-warp nms-padding)
all_size=(1000 2000 5000 8000 10000)

for file in "${all_files[@]}"; do
    for size in "${all_size[@]}"; do
        echo
        echo "========================================"
        echo "[BATCH] file=$file  n=$size"
        echo "========================================"
        echo

        ./run.sh "$file" "$size"

        echo
        echo "[BATCH DONE] file=$file n=$size"
        echo
    done
done