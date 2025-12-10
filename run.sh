#!/bin/bash
EXEC=$1
shift

echo "===== Warm Up ====="
./"$EXEC" "$@"

echo
echo "===== Profiling ====="
nvprof ./"$EXEC" "$@"

srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof --print-gpu-summary --log-file time_distribution.log \
--metrics achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput \
./"$EXEC" "$@"