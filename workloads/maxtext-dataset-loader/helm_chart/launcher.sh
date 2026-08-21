#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

echo "=== MaxText Dataset Loading Benchmark Launcher ==="
echo "LOADER_MODE: ${LOADER_MODE:-storage_bench}"
echo "DATASET_PATH: ${DATASET_PATH}"
echo "ACCESS_MODE: ${ACCESS_MODE:-auto}"
echo "SHUFFLE_MODE: ${SHUFFLE_MODE:-none}"
echo "DATASET_FORMAT: ${DATASET_FORMAT:-parquet}"
echo "CONVERT_TO_ARRAYRECORD: ${CONVERT_TO_ARRAYRECORD:-false}"
echo "COLUMNS: ${COLUMNS:-input_ids,label}"
echo "BATCH_SIZE: ${BATCH_SIZE:-64}"
echo "MAX_BATCHES: ${MAX_BATCHES:-100}"
echo "NUM_THREADS: ${NUM_THREADS:-4}"
echo "GENERATE_DATASET: ${GENERATE_DATASET:-false}"

if [ -f "/workload/requirements.txt" ]; then
  echo "--> Installing requirements from /workload/requirements.txt..."
  pip3 install --no-cache-dir -r /workload/requirements.txt || true
fi

if [ "${GENERATE_DATASET}" = "true" ]; then
  if [ -z "${DATASET_PATH}" ]; then
    echo "ERROR: DATASET_PATH is required for dataset generation!"
    exit 1
  fi
  echo "--> Generating synthetic MaxText Parquet dataset at ${DATASET_PATH}..."
  python3 /workload/dataset_generator.py \
    --output-path="${DATASET_PATH}" \
    --total-size-mb="${DATASET_SIZE_MB:-1024}" \
    --num-files="${NUM_FILES:-10}" \
    --sequence-length="${SEQUENCE_LENGTH:-2048}" \
    --metadata-bytes-per-row="${METADATA_BYTES:-4096}"
fi

if [ "${CONVERT_TO_ARRAYRECORD}" = "true" ]; then
  echo "--> Converting Parquet dataset at ${DATASET_PATH} to ArrayRecord dataset..."
  python3 /workload/parquet_to_arrayrecord.py \
    --input-path="${DATASET_PATH}" \
    --output-path="${DATASET_PATH}/arrayrecord_dataset" \
    --sequence-length="${SEQUENCE_LENGTH:-2048}" \
    --max-files="${NUM_FILES:-0}" \
    --num-workers="${NUM_THREADS:-32}"
fi

if [ "${LOADER_MODE}" = "in_tree_loader" ]; then
  echo "--> Executing MaxText In-Tree Standalone DataLoader Benchmark (format=${DATASET_FORMAT:-auto})..."
  export JAX_PLATFORMS="${HARDWARE:-cpu}"
  BENCH_PATH="${DATASET_PATH}"
  if [ "${DATASET_FORMAT}" = "arrayrecord" ] && [[ "${DATASET_PATH}" != *"arrayrecord"* ]] && [ -d "${DATASET_PATH}/arrayrecord_dataset" ]; then
    BENCH_PATH="${DATASET_PATH}/arrayrecord_dataset"
  fi
  python3 /workload/standalone_dataloader.py \
    --config-path="${MAXTEXT_CONFIG_PATH:-src/maxtext/configs/base.yml}" \
    --run-name="${MAXTEXT_RUN_NAME:-maxtext_in_tree_bench}" \
    --dataset-type="${MAXTEXT_DATASET_TYPE:-synthetic}" \
    --dataset-path="${BENCH_PATH}" \
    --dataset-format="${DATASET_FORMAT:-auto}" \
    --use-manifest="${USE_MANIFEST:-true}" \
    --shuffle-mode="${SHUFFLE_MODE:-two_stage}" \
    --shuffle-buffer-size="${SHUFFLE_BUFFER_SIZE:-20000}" \
    --num-streams="${NUM_THREADS:-8}" \
    --steps="${MAX_BATCHES:-500}" \
    --per-device-batch-size="${BATCH_SIZE:-128}" \
    --sequence-length="${SEQUENCE_LENGTH:-2048}" \
    --chunk-records="${CHUNK_RECORDS:-1}" \
    --hardware="${HARDWARE:-cpu}"
else
  if [ -z "${DATASET_PATH}" ]; then
    echo "ERROR: DATASET_PATH is required!"
    exit 1
  fi

  BENCH_PATH="${DATASET_PATH}"
  if [ "${DATASET_FORMAT}" = "arrayrecord" ] && [[ "${DATASET_PATH}" != *"arrayrecord"* ]] && [ -d "${DATASET_PATH}/arrayrecord_dataset" ]; then
    BENCH_PATH="${DATASET_PATH}/arrayrecord_dataset"
  fi

  echo "--> Executing MaxText Dataset Benchmark (format=${DATASET_FORMAT:-parquet})..."
  python3 /workload/maxtext_dataset_bench.py \
    --dataset-path="${BENCH_PATH}" \
    --format="${DATASET_FORMAT:-parquet}" \
    --access-mode="${ACCESS_MODE:-auto}" \
    --shuffle-mode="${SHUFFLE_MODE:-none}" \
    --columns="${COLUMNS:-input_ids,label}" \
    --batch-size="${BATCH_SIZE:-64}" \
    --max-batches="${MAX_BATCHES:-100}" \
    --num-threads="${NUM_THREADS:-4}" \
    --num-streams="${NUM_STREAMS:-${NUM_THREADS:-4}}" \
    --rank="${RANK:-0}" \
    --world-size="${WORLD_SIZE:-1}"
fi

echo "=== MaxText Benchmark Finished Successfully ==="
