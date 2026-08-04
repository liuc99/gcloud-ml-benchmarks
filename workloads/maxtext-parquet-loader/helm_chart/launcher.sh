#!/bin/bash
set -e

echo "=== MaxText Parquet Dataset Loading & GCS Range Read Launcher ==="
echo "DATASET_PATH: ${DATASET_PATH}"
echo "ACCESS_MODE: ${ACCESS_MODE:-auto}"
echo "COLUMNS: ${COLUMNS:-input_ids,label}"
echo "BATCH_SIZE: ${BATCH_SIZE:-64}"
echo "MAX_BATCHES: ${MAX_BATCHES:-100}"
echo "NUM_THREADS: ${NUM_THREADS:-4}"
echo "GENERATE_DATASET: ${GENERATE_DATASET:-false}"

if [ -z "${DATASET_PATH}" ]; then
  echo "ERROR: DATASET_PATH is required!"
  exit 1
fi

if [ -f "/workload/requirements.txt" ]; then
  echo "--> Installing requirements from /workload/requirements.txt..."
  pip3 install --no-cache-dir -r /workload/requirements.txt
fi

if [ "${GENERATE_DATASET}" = "true" ]; then
  echo "--> Generating synthetic MaxText Parquet dataset at ${DATASET_PATH}..."
  python3 /workload/dataset_generator.py \
    --output-path="${DATASET_PATH}" \
    --total-size-mb="${DATASET_SIZE_MB:-1024}" \
    --num-files="${NUM_FILES:-10}" \
    --sequence-length="${SEQUENCE_LENGTH:-2048}" \
    --metadata-bytes-per-row="${METADATA_BYTES:-4096}"
fi

echo "--> Executing MaxText Parquet GCS Range Read Benchmark..."
python3 /workload/maxtext_parquet_bench.py \
  --dataset-path="${DATASET_PATH}" \
  --access-mode="${ACCESS_MODE:-auto}" \
  --columns="${COLUMNS:-input_ids,label}" \
  --batch-size="${BATCH_SIZE:-64}" \
  --max-batches="${MAX_BATCHES:-100}" \
  --num-threads="${NUM_THREADS:-4}" \
  --rank="${RANK:-0}" \
  --world-size="${WORLD_SIZE:-1}"

echo "=== MaxText Parquet Benchmark Finished Successfully ==="
