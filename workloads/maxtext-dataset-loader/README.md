# MaxText Dataset Loader & Storage Benchmark (`maxtext-dataset-loader`)

A cloud-native benchmark workload and demo harness for evaluating large-scale LLM dataset loading pipelines on Google Cloud Platform (GCS / GCSFuse CSI / Persistent Disk / Managed Lustre), comparing:
1. **ArrayRecord (Pre-Tokenized Binary)** vs. **Parquet (Raw Columnar Text)**
2. **Storage Access Modes**: GCSFuse CSI Driver Mount (`accessMode=gcsfuse`) vs. Direct GCS Client (`accessMode=native_gcs` / `gcsfs`)
3. **Execution Modes**:
   - `loaderMode=storage_bench`: Raw storage I/O, column projection, and GCS range-read micro-benchmarks.
   - `loaderMode=in_tree_loader`: End-to-end MaxText standalone DataLoader simulating PyGrain two-stage shuffle, buffer priming, and runtime CPU Tokenizer profiling.

---

## 🚀 Key Performance Highlights (1,650 Shards / 420 GB Dataset)

| Evaluation Dimension | Parquet (Un-tokenized) | ArrayRecord (Pre-tokenized) | Performance Advantage |
| :--- | :--- | :--- | :--- |
| **Storage & Transfer Size** | **420.10 GB** (100%) | **155.27 GB** (**36.1%**) | **63.9% reduction in storage cost and network transfer** |
| **Training CPU Overhead** | High (79% ~ 92% of step duration) | **Zero-CPU Ingestion** | Eliminates host CPU tokenization bottleneck |
| **Cold Start / TTFB** | **7.28 s** (20,000 sample buffer priming) | **3.53 s** (Direct binary read) | **51.5% reduction in startup stall** |
| **Steady-State Step Latency (Avg)** | **23.76 ms** (With 8-thread tokenizer) | **19.15 ms** (Direct tensor mapping) | **19.4% reduction in per-step batch latency** |
| **Effective Tensor Throughput** | **3,287 samples/s** (25.68 MB/s) | **4,888 samples/s** (38.19 MB/s) | **+48.7% higher effective ingestion throughput** |

> 📊 For full empirical data, charts, and analysis, see [Parquet vs. ArrayRecord Results](../../docs/maxtext/results/parquet_vs_arrayrecord.md) and [Shuffle Strategies Breakdown](../../docs/maxtext/results/shuffle_strategies.md).

---

## 🛠️ CLI Standalone Converters & Generators

### 1. Multi-Process Parquet to ArrayRecord Converter
Converts multi-column Parquet shards into pre-tokenized `.array_record` binary shards using `ProcessPoolExecutor` to saturate available host vCPUs:

```bash
python3 workloads/maxtext-dataset-loader/helm_chart/parquet_to_arrayrecord.py \
  --input-path="gs://my-bucket/parquet_dataset" \
  --output-path="gs://my-bucket/arrayrecord_dataset" \
  --sequence-length=2048 \
  --max-files=20
```

### 2. Synthetic Dataset Generator
Generates sample multi-column Parquet datasets directly on GCS:

```bash
python3 workloads/maxtext-dataset-loader/helm_chart/dataset_generator.py \
  --output-path="gs://my-bucket/maxtext_parquet_dataset" \
  --total-size-mb=10240 \
  --num-files=20 \
  --sequence-length=2048 \
  --metadata-bytes-per-row=4096
```

---

## ☸️ Helm Chart Parameters Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `gcsfuse.enabled` | bool | `true` | Enables GCSFuse CSI Driver sidecar mount. |
| `gcsfuse.datasetBucket` | string | `""` | Target GCS bucket containing datasets. |
| `gcsfuse.mountOptions` | string | `"implicit-dirs,stat-cache-capacity:1000000..."` | GCSFuse mount options (caching & buffers). |
| `gcsfs.datasetPath` | string | `"/gcs/dataset"` | Dataset path inside container (`/gcs/...` or `gs://...`). |
| `workload.accessMode` | string | `"gcsfuse"` | Storage access mode (`gcsfuse` or `native_gcs`). |
| `workload.loaderMode` | string | `"in_tree_loader"` | Execution mode (`in_tree_loader` or `storage_bench`). |
| `workload.datasetFormat` | string | `"arrayrecord"` | Target dataset format (`arrayrecord` or `parquet`). |
| `workload.shuffleMode` | string | `"two_stage"` | Shuffling strategy (`none`, `two_stage`, or `global`). |
| `workload.shuffleBufferSize` | int | `20000` | Sliding window sample buffer size for two-stage shuffle. |
| `workload.numStreams` | int | `8` | Number of concurrent interleaved shard streams. |
| `workload.batchSize` | int | `128` | Per-step batch size. |
| `workload.maxBatches` | int | `500` | Total number of batches to benchmark. |
| `nodeSelector` | map | `{"cloud.google.com/gke-nodepool": "n4-standard-80"}` | Kubernetes node selector for compute pods. |

---

## 🚀 Quickstart Deployment

### Scenario A: In-Tree Standalone DataLoader Benchmark (Two-Stage Shuffle)
```bash
helm install maxtext-intree-run workloads/maxtext-dataset-loader/helm_chart \
  --set gcsfuse.enabled=true \
  --set gcsfuse.datasetBucket="<YOUR_GCS_BUCKET>" \
  --set gcsfs.datasetPath="/gcs/dataset" \
  --set workload.accessMode="gcsfuse" \
  --set workload.loaderMode="in_tree_loader" \
  --set workload.datasetFormat="arrayrecord" \
  --set workload.shuffleMode="two_stage" \
  --set workload.batchSize=128 \
  --set workload.maxBatches=500 \
  --set nodeSelector."cloud\.google\.com/gke-nodepool"="n4-standard-80"
```

### Scenario B: Raw Storage Range-Read Benchmark
```bash
helm install maxtext-storage-run workloads/maxtext-dataset-loader/helm_chart \
  --set gcsfuse.enabled=true \
  --set gcsfuse.datasetBucket="<YOUR_GCS_BUCKET>" \
  --set gcsfs.datasetPath="/gcs/dataset" \
  --set workload.accessMode="gcsfuse" \
  --set workload.loaderMode="storage_bench" \
  --set workload.datasetFormat="parquet" \
  --set workload.batchSize=128 \
  --set workload.maxBatches=500
```

---

## 📚 Complete Documentation Suite

- [MaxText Documentation Index](../../docs/maxtext/README.md)
- [Parquet vs. ArrayRecord Results](../../docs/maxtext/results/parquet_vs_arrayrecord.md)
- [Shuffle Strategies: None vs. Two-Stage vs. Global](../../docs/maxtext/results/shuffle_strategies.md)
- [Storage Access Modes: GCSFuse CSI vs. Native GCS](../../docs/maxtext/results/storage_access_modes.md)
- [Parquet Range Reads & ArrayRecord Guide](../../docs/maxtext/parquet_range_reads_guide.md)
