# Multi-Format ML Dataset Loading Benchmark (`multi-format-dataset-loader`)

A high-performance benchmark workload for evaluating dataset streaming throughput, Time-to-First-Batch (TTFB), and sample ingestion latency across **Parquet**, **WebDataset TAR**, **Zarr / TensorStore**, **PyTorch `.pt`**, and **JSONL** on Google Cloud Storage and Google Cloud Managed Lustre.

---

## 📋 Supported Formats & Readers Matrix

| Dataset Format | Recommended Reader Framework | Storage Backend Tested | Typical Use Case |
| :--- | :--- | :--- | :--- |
| **Parquet** | HuggingFace `datasets.load_dataset(..., streaming=True)` | GCSFuse, `gcsfs`, Managed Lustre | LLM Pre-training / Text & Token Sequences |
| **ArrayRecord** | C++ `ArrayRecordReader` / `array-record` | Native GCS, GCSFuse, Staging | Pre-tokenized LLM Token Array Sequences (MaxText / JAX) |
| **WebDataset TAR** | `webdataset.WebLoader` | GCSFuse, Managed Lustre | Vision, Audio & Multimodal Datasets |
| **Zarr / TensorStore** | `tensorstore.open(...)` | GCS Native KVStore, GCSFuse, Lustre | Multimodal Tensors, Medical Imaging, Climate Models |
| **PyTorch `.pt`** | PyTorch `DataLoader` + `gcsfs` / POSIX | GCSFuse, `gcsfs`, Managed Lustre | Pre-processed Tensor Shards |
| **JSONL** | Custom IterableDataset / PyArrow | GCSFuse, `gcsfs`, Managed Lustre | Lightweight Text / Document Shards |

> 📊 For detailed benchmark findings and performance comparisons across formats, see [Multi-Format Performance Results](../../docs/multi_format_dataset/results/format_comparison.md).

---

## 🛠️ CLI Standalone Tools

### 1. Synthetic Dataset Generator
```bash
python3 workloads/multi-format-dataset-loader/helm_chart/dataset_generator.py \
  --output-path="gs://my-bucket/bench_dataset_parquet" \
  --format=parquet \
  --total-size-mb=10240 \
  --num-files=50
```

### 2. Standalone Dataset Loading Benchmark
```bash
python3 workloads/multi-format-dataset-loader/helm_chart/dataset_loading_bench.py \
  --dataset-path="/gcs/my-bucket/bench_dataset_parquet" \
  --format=parquet \
  --reader=hf_datasets \
  --batch-size=64 \
  --num-workers=8 \
  --max-batches=500
```

---

## ☸️ Helm Chart Parameters Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `workload.nodes` | int | `1` | Number of distributed worker nodes (JobSet replicated pods). |
| `workload.format` | string | `"parquet"` | Dataset format (`parquet`, `webdataset`, `zarr`, `torch`, `jsonl`). |
| `workload.reader` | string | `"hf_datasets"` | Loader framework (`hf_datasets`, `webdataset`, `tensorstore`, `torch_dataloader`). |
| `workload.batchSize` | int | `64` | Ingestion batch size per worker. |
| `workload.numWorkers` | int | `8` | Data loading worker processes per pod (`DataLoader(num_workers=8)`). |
| `workload.maxBatches` | int | `500` | Total batches to consume for benchmark run. |
| `gcsfuse.enabled` | bool | `false` | Enables GCSFuse CSI Driver sidecar mount. |
| `gcsfuse.datasetBucket` | string | `""` | GCS Bucket to mount at `/gcs`. |
| `lustre.enabled` | bool | `false` | Enables Managed Lustre PVC mount at `/lustre`. |
| `lustre.checkpointPvc` | string | `""` | PersistentVolumeClaim name for Managed Lustre. |
| `gcsfs.datasetPath` | string | `""` | Dataset path (`/gcs/...`, `/lustre/...`, or `gs://...`). |

---

## 🚀 Quickstart Deployment

### Scenario 1: GCSFuse Streaming Reads (CSI Sidecar Mount)
```bash
helm install bench-dataset-gcsfuse workloads/multi-format-dataset-loader/helm_chart -f workloads/multi-format-dataset-loader/helm_chart/values_base.yaml \
  --set workload.nodes=2 \
  --set workload.format="parquet" \
  --set workload.reader="hf_datasets" \
  --set workload.batchSize=64 \
  --set workload.numWorkers=8 \
  --set workload.maxBatches=500 \
  --set gcsfuse.enabled=true \
  --set gcsfuse.datasetBucket="<YOUR_BUCKET>" \
  --set-string gcsfuse.mountOptions="implicit-dirs\,file-cache:max-size-mb:-1\,file-cache:cache-file-for-range-read:true" \
  --set gcsfs.datasetPath="/gcs/dataset/bench_dataset_parquet"
```

### Scenario 2: Direct GCS Client (`gcsfs` / Native REST)
```bash
helm install bench-dataset-gcsfs workloads/multi-format-dataset-loader/helm_chart -f workloads/multi-format-dataset-loader/helm_chart/values_base.yaml \
  --set workload.nodes=2 \
  --set workload.format="parquet" \
  --set workload.reader="hf_datasets" \
  --set workload.batchSize=64 \
  --set workload.numWorkers=8 \
  --set workload.maxBatches=500 \
  --set gcsfs.datasetPath="gs://<YOUR_BUCKET>/bench_dataset_parquet"
```

### Scenario 3: Google Cloud Managed Lustre
```bash
helm install bench-dataset-lustre workloads/multi-format-dataset-loader/helm_chart -f workloads/multi-format-dataset-loader/helm_chart/values_base.yaml \
  --set workload.nodes=2 \
  --set workload.format="parquet" \
  --set workload.reader="hf_datasets" \
  --set workload.batchSize=64 \
  --set workload.numWorkers=8 \
  --set workload.maxBatches=500 \
  --set lustre.enabled=true \
  --set lustre.checkpointPvc="lustre-pvc" \
  --set gcsfs.datasetPath="/lustre/dataset/bench_dataset_parquet"
```

---

## 📚 Complete Documentation Suite

- [Multi-Format Dataset Documentation Index](../../docs/multi_format_dataset/README.md)
- [Step-by-Step Reproduction Guide](../../docs/multi_format_dataset/step_by_step_guide.md)
- [Format & Backend Performance Comparison](../../docs/multi_format_dataset/results/format_comparison.md)
