#!/bin/bash
# Orbax Checkpoint Resharding and Restore Workload Launcher
set -euo pipefail

export PYTHONUNBUFFERED=1

echo "================================================================="
echo " Starting Orbax Checkpoint Resharding & Restore Benchmark Launcher "
echo "================================================================="

# Install Python requirements
STEP_START=$(date +%s)
echo "Installing Python dependencies..."
pip3 install --no-cache-dir -r /workload/configs/requirements.txt

if [[ -n "${REQUIREMENTS:-}" ]] && [[ "${REQUIREMENTS:-}" != "none" ]]; then
  pip3 install $REQUIREMENTS
fi
echo "[BENCHMARK] Python dependencies setup finished in $(( $(date +%s) - STEP_START )) seconds."

# Determine mount path
MOUNT_PATH="${CKPT_WRITE_PATH:-/gcs/checkpoints}"
if [[ "$MOUNT_PATH" == gs://* ]]; then
  echo "Warning: CKPT_WRITE_PATH is $MOUNT_PATH, defaulting to /gcs/checkpoints for GCSFuse mount."
  MOUNT_PATH="/gcs/checkpoints"
fi

echo "Running Orbax benchmark against mount path: $MOUNT_PATH"

BENCHMARK_MODE="${BENCHMARK_MODE:-compare}"
SRC_SHARDS="${SRC_SHARDS:-5}"
DST_WORKERS="${DST_WORKERS:-10}"
NUM_LAYERS="${NUM_LAYERS:-4}"
HIDDEN_DIM="${HIDDEN_DIM:-4096}"
STRATEGY="${STRATEGY:-dim_partitions}"
DIM_PARTITIONS="${DIM_PARTITIONS:-0:10}"
TARGET_CHUNK_MB="${TARGET_CHUNK_MB:-64.0}"
CAST_DTYPE="${CAST_DTYPE:-keep}"
NUM_RUNS="${NUM_RUNS:-5}"

NODE_RANK="${JOB_COMPLETION_INDEX:-${NODE_RANK:-0}}"
NUM_NODES="${NNODES:-${NODES:-1}}"

EXTRA_ARGS=()
if [[ "${STRIP_OPT_STATE:-false}" == "true" ]] || [[ "${STRIP_OPT_STATE:-false}" == "1" ]]; then
  EXTRA_ARGS+=("--strip-opt-state")
fi

echo "Node Rank: $NODE_RANK / $NUM_NODES | Source Shards: $SRC_SHARDS -> Target Workers: $DST_WORKERS"

python3 -u /workload/orbax_checkpoint_bench.py \
  --mount-path "$MOUNT_PATH" \
  --mode "$BENCHMARK_MODE" \
  --src-shards "$SRC_SHARDS" \
  --dst-workers "$DST_WORKERS" \
  --num-layers "$NUM_LAYERS" \
  --hidden-dim "$HIDDEN_DIM" \
  --strategy "$STRATEGY" \
  --dim-partitions "$DIM_PARTITIONS" \
  --target-chunk-mb "$TARGET_CHUNK_MB" \
  --cast-dtype "$CAST_DTYPE" \
  --num-runs "$NUM_RUNS" \
  --node-rank "$NODE_RANK" \
  --num-nodes "$NUM_NODES" \
  "${EXTRA_ARGS[@]}"

echo "Orbax Checkpoint Benchmark completed successfully on node $NODE_RANK."
