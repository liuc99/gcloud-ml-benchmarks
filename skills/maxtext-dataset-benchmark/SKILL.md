---
name: maxtext-dataset-benchmark
description: Workload sub-skill for executing MaxText dataset loading benchmarks (Parquet GCS Range Reads vs ArrayRecord pre-tokenized streaming), running the Parquet-to-ArrayRecord converter, and executing shuffle strategy evaluations on GKE.
---

# MaxText Dataset Benchmark & Conversion Sub-Skill (`maxtext-dataset-benchmark`)

This is a workload sub-skill for executing MaxText dataset loading benchmarks on GKE. It handles the technical execution of **Parquet GCS Range Reads**, **ArrayRecord pre-tokenized streaming**, the **`parquet_to_arrayrecord.py` converter**, and **Shuffle strategy (`none`, `two_stage`, `global`)** benchmark runs.

*Note*: Master interactive questionnaires and Plan Approval Protocols are governed by [`ml-benchmark-orchestrator`](../ml-benchmark-orchestrator/SKILL.md).

---

## 🛠️ Technical Execution Protocols

### 1. Optional Parquet to ArrayRecord Preprocessing
When converting raw Parquet text shards to pre-tokenized `.array_record` shards holding `int32` token arrays:

```bash
python3 workloads/maxtext-parquet-loader/helm_chart/parquet_to_arrayrecord.py \
  --input-path="${INPUT_PARQUET_PATH}" \
  --output-path="${OUTPUT_ARRAYRECORD_PATH}" \
  --sequence-length=${SEQUENCE_LENGTH:-2048} \
  --max-files=${MAX_FILES:-20}
```

---

### 2. Deploy Benchmark Runs via Helm

Navigate to `workloads/maxtext-parquet-loader/helm_chart`.

#### Run A: Parquet Loader Benchmark (`format=parquet`)
```bash
helm install maxtext-demo-run . \
  --set gcsfs.datasetPath="${DATASET_PATH}" \
  --set workload.datasetFormat="parquet" \
  --set workload.shuffleMode="${SHUFFLE_MODE:-two_stage}" \
  --set workload.batchSize=${BATCH_SIZE:-64} \
  --set workload.maxBatches=${MAX_BATCHES:-100}
```

#### Run B: ArrayRecord Loader Benchmark (`format=arrayrecord`)
```bash
helm install maxtext-demo-run . \
  --set gcsfs.datasetPath="${ARRAYRECORD_DATASET_PATH}" \
  --set workload.datasetFormat="arrayrecord" \
  --set workload.convertToArrayRecord=false \
  --set workload.shuffleMode="${SHUFFLE_MODE:-two_stage}" \
  --set workload.batchSize=${BATCH_SIZE:-64} \
  --set workload.maxBatches=${MAX_BATCHES:-100}
```

---

### 3. Shuffle Strategy Comparative Evaluation Protocol

When comparing shuffle strategies (`none`, `two_stage`, `global`):

```bash
for SHUFFLE in "none" "two_stage" "global"; do
  echo ">>> Running Benchmark: format=${FORMAT}, shuffle=${SHUFFLE}"
  helm install maxtext-demo-run . \
    --set gcsfs.datasetPath="${DATASET_PATH}" \
    --set workload.datasetFormat="${FORMAT}" \
    --set workload.convertToArrayRecord=false \
    --set workload.shuffleMode="${SHUFFLE}" \
    --set workload.batchSize=${BATCH_SIZE:-64} \
    --set workload.maxBatches=${MAX_BATCHES:-100}

  # Monitor pod completion
  kubectl wait --for=condition=Ready pod -l jobset.sigs.k8s.io/jobset-name=maxtext-demo-run --timeout=120s
  
  # Collect logs & summary metrics
  kubectl logs -l jobset.sigs.k8s.io/jobset-name=maxtext-demo-run -c workload --tail=30
  
  # Uninstall before next iteration
  helm uninstall maxtext-demo-run
done
```

---

### 4. Milestone Monitoring & Teardown
- Monitor Pod status: `kubectl get pods -l jobset.sigs.k8s.io/jobset-name=maxtext-demo-run`
- Fetch summary metrics: `kubectl logs pod/${POD_NAME} --tail=40`
- Teardown: `helm uninstall maxtext-demo-run --ignore-not-found`
