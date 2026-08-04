---
name: maxtext-dataset-benchmark
description: Skill for converting Parquet datasets to ArrayRecord, deploying MaxText dataset loading benchmarks on GKE, and running Parquet vs ArrayRecord & Shuffle strategy comparative benchmarks.
---

# MaxText Dataset Benchmark & Conversion Skill (`maxtext-dataset-benchmark`)

This skill enables an AI Agent to automate dataset conversion from Parquet to ArrayRecord, deploy MaxText dataset loading benchmarks on GKE via Helm, run multi-strategy shuffle benchmarks (`none`, `two_stage`, `global`), and parse ingestion metrics.

---

## 🛠️ Tasks & Capabilities

### 1. Parquet to ArrayRecord Dataset Conversion
Convert raw Parquet text shards to pre-tokenized `.array_record` shards holding `int32` token arrays using the standalone converter:

```bash
python3 workloads/maxtext-parquet-loader/helm_chart/parquet_to_arrayrecord.py \
  --input-path="${INPUT_PARQUET_PATH}" \
  --output-path="${OUTPUT_ARRAYRECORD_PATH}" \
  --sequence-length=${SEQUENCE_LENGTH:-2048} \
  --max-files=${MAX_FILES:-20}
```

*Note*: Supports GCS RAPID zonal storage class (`gs://...`) using `gcsfs` appendable streaming writes.

---

### 2. Deploy Benchmark Runs via Helm

Navigate to `workloads/maxtext-parquet-loader/helm_chart`.

#### Run A: Parquet Loader Benchmark (`parquet`)
```bash
helm install maxtext-demo-run . \
  --set gcsfs.datasetPath="${DATASET_PATH}" \
  --set workload.datasetFormat="parquet" \
  --set workload.shuffleMode="${SHUFFLE_MODE:-two_stage}" \
  --set workload.batchSize=64 \
  --set workload.maxBatches=100
```

#### Run B: ArrayRecord Loader Benchmark (`arrayrecord`)
```bash
helm install maxtext-demo-run . \
  --set gcsfs.datasetPath="${DATASET_PATH}" \
  --set workload.datasetFormat="arrayrecord" \
  --set workload.convertToArrayRecord=false \
  --set workload.shuffleMode="${SHUFFLE_MODE:-two_stage}" \
  --set workload.batchSize=64 \
  --set workload.maxBatches=100
```

---

### 3. Asynchronous Log & Milestone Monitoring
Monitor pod execution and log summary output:

```bash
# Get pod status
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=maxtext-demo-run

# Stream/Fetch benchmark output
kubectl logs pod/${POD_NAME} --tail=40
```

Verify key metrics in output:
- **Time to First Batch (TTFB)** (ms)
- **Upfront Index Scanning Penalty** (ms)
- **Batch Load Latency Percentiles** (`p50`, `p95`, `p99` ms)
- **Sample Ingestion Speed** (samples/sec)
- **Payload Data Read Volume** (MB)

---

### 4. Release Cleanup & Teardown
Uninstall the Helm release when benchmark runs complete:

```bash
helm uninstall maxtext-demo-run --ignore-not-found
```
