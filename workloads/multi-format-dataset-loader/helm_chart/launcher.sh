#!/bin/bash
set -e

echo "=== ML Dataset Loading Benchmark & Demo Launcher ==="
echo "DATASET_PATH: ${DATASET_PATH}"
echo "FORMAT: ${FORMAT:-parquet}"
echo "READER: ${READER:-hf_datasets}"
echo "BATCH_SIZE: ${BATCH_SIZE:-64}"
echo "NUM_WORKERS: ${NUM_WORKERS:-4}"
echo "PREFETCH_FACTOR: ${PREFETCH_FACTOR:-2}"
echo "MAX_BATCHES: ${MAX_BATCHES:-100}"
echo "GENERATE_DATASET: ${GENERATE_DATASET:-false}"

if [ -z "${DATASET_PATH}" ]; then
  echo "ERROR: DATASET_PATH is required!"
  exit 1
fi

if [ -f "/workload/requirements.txt" ]; then
  echo "--> Installing requirements from /workload/requirements.txt..."
  pip3 install --no-cache-dir -r /workload/requirements.txt
fi

# Optional: Generate synthetic dataset if requested or directory is missing
if [ "${GENERATE_DATASET}" = "true" ]; then
  echo "--> Generating synthetic ${FORMAT} dataset at ${DATASET_PATH}..."
  python3 /workload/dataset_generator.py \
    --output-path="${DATASET_PATH}" \
    --format="${FORMAT:-parquet}" \
    --total-size-mb="${DATASET_SIZE_MB:-1024}" \
    --num-files="${NUM_FILES:-10}" \
    --sequence-length="${SEQUENCE_LENGTH:-512}" \
    --metadata-bytes-per-row="${METADATA_BYTES:-4096}"
fi

USE_MANIFEST_FLAG=""
if [ "${USE_MANIFEST}" = "true" ]; then
  USE_MANIFEST_FLAG="--use-manifest"
fi

echo "--> Executing Dataset Loading Benchmark..."
python3 /workload/dataset_loading_bench.py \
  --dataset-path="${DATASET_PATH}" \
  --format="${FORMAT:-parquet}" \
  --reader="${READER:-hf_datasets}" \
  --batch-size="${BATCH_SIZE:-64}" \
  --num-workers="${NUM_WORKERS:-4}" \
  --prefetch-factor="${PREFETCH_FACTOR:-2}" \
  --max-batches="${MAX_BATCHES:-100}" \
  --epochs="${EPOCHS:-1}" \
  --rank="${RANK:-0}" \
  --world-size="${WORLD_SIZE:-1}" \
  --shuffle-strategy="${SHUFFLE_STRATEGY:-none}" \
  --buffer-size="${BUFFER_SIZE:-10000}" \
  --seed="${SEED:-42}" \
  ${USE_MANIFEST_FLAG}

echo "=== Dataset Loading Benchmark Finished Successfully ==="
