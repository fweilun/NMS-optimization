#!/bin/bash
ml cuda

EXEC=$1
shift
make $EXEC

# Create profiles directory if it doesn't exist
mkdir -p profiles

echo "===== Warm Up ====="
srun -p nvidia -N1 -n1 --gres=gpu:1 ./"$EXEC" "$@"

# echo
echo "===== Profiling ====="

srun -p nvidia -N1 -n1 --gres=gpu:1 \
nvprof --metrics achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput \
--log-file profiles/profiling_metrics-\($EXEC-$@\).log ./"$EXEC" "$@"

srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof --print-gpu-summary --log-file time_distribution.log \
 --csv --log-file profiles/time_distribution-\($EXEC-$@\).csv ./"$EXEC" "$@"