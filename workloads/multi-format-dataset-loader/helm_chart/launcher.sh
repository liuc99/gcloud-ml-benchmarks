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

# Optional: Generate synthetic dataset if requested or directory is missing
if [ "${GENERATE_DATASET}" = "true" ]; then
  echo "--> Generating synthetic ${FORMAT} dataset at ${DATASET_PATH}..."
  python3 /workload/dataset_generator.py \
    --output-path="${DATASET_PATH}" \
    --format="${FORMAT:-parquet}" \
    --total-size-mb="${DATASET_SIZE_MB:-1024}" \
    --num-files="${NUM_FILES:-10}" \
    --sequence-length="${SEQUENCE_LENGTH:-512}" \
    --embedding-dim="${EMBEDDING_DIM:-768}"
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
  --world-size="${WORLD_SIZE:-1}"

echo "=== Dataset Loading Benchmark Finished Successfully ==="
