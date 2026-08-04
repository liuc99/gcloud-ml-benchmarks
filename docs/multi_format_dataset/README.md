# ML Dataset Loading Benchmarks & Demos

A dedicated benchmarking and PoC framework within `gcloud-ml-benchmarks` for evaluating high-throughput ML dataset streaming, batch ingestion speed, and sample access latency on Google Cloud Platform (GCP) and Google Kubernetes Engine (GKE).

---

## 🎯 Key Features

1. **Synthetic Dataset Generator (`dataset_generator.py`)**:
   - Generates benchmark datasets directly on **GCS (`gs://...`)**, **GCSFuse mounts (`/gcs/...`)**, **Google Cloud Managed Lustre (`/lustre/...`)**, or local disk.
   - Supported Formats:
     - **Parquet** (HuggingFace streaming Parquet files)
     - **WebDataset TAR** (Shard-based TAR archives containing features & JSON metadata)
     - **Zarr / TensorStore** (Multi-dimensional array chunks)
     - **PyTorch `.pt`** (Raw PyTorch tensor shards)
     - **JSONL** (Lightweight text/document sequence shards)

2. **Dataset Loading Benchmark Harness (`dataset_loading_bench.py`)**:
   - Evaluates key data loader frameworks: **HuggingFace `datasets`**, **WebDataset**, **TensorStore**, **PyTorch `DataLoader`**, and **Native PyArrow / GCSFS**.
   - Collects core performance metrics:
     - ⏱️ **Time to First Batch (TTFB)** (ms / sec)
     - 🚀 **Read Throughput** (MB/s and Gbps)
     - ⚡ **Ingestion Speed** (samples/sec)
     - 📊 **Batch Latency Percentiles** (p50, p95, p99 ms)
     - 👥 **Multi-Worker & Multi-Node Scaling** (`num_workers`, `prefetch_factor`, DDP rank sharding)

3. **Helm & Kubernetes JobSet Integration**:
   - Easily deploy single-node and multi-node dataset loading benchmark releases on GKE using Helm and Kubernetes JobSet.

---

## 📊 Supported Formats & Readers Matrix

| Dataset Format | Recommended Reader Framework | Storage Backend Tested | Typical Use Case |
| :--- | :--- | :--- | :--- |
| **Parquet** | HuggingFace `datasets.load_dataset(..., streaming=True)` | GCSFuse, `gcsfs`, Managed Lustre | LLM Pre-training / Text & Token Sequences |
| **ArrayRecord** | C++ `ArrayRecordReader` / `array-record` | Native GCS, GCSFuse, Staging | Pre-tokenized LLM Token Array Sequences (MaxText / JAX) |
| **WebDataset TAR** | `webdataset.WebLoader` | GCSFuse, Managed Lustre | Vision, Audio & Multimodal Datasets |
| **Zarr / TensorStore** | `tensorstore.open(...)` | GCS Native KVStore, GCSFuse, Lustre | Multimodal Tensors, Medical Imaging, Climate Models |
| **PyTorch `.pt`** | PyTorch `DataLoader` + `gcsfs` / POSIX | GCSFuse, `gcsfs`, Managed Lustre | Pre-processed Tensor Shards |
| **JSONL** | Custom IterableDataset / PyArrow | GCSFuse, `gcsfs`, Managed Lustre | Lightweight Text / Document Shards |

---

## 🚀 Quickstart: Local & Demo Usage

Generate a sample 1 GB Parquet dataset locally and run the benchmark:

```bash
# 1. Generate 1 GB Parquet dataset (10 shards)
python3 workloads/multi-format-dataset-loader/helm_chart/dataset_generator.py \
  --output-path=/tmp/demo_dataset \
  --format=parquet \
  --total-size-mb=1024 \
  --num-files=10

# 2. Run Dataset Loading Benchmark
python3 workloads/multi-format-dataset-loader/helm_chart/dataset_loading_bench.py \
  --dataset-path=/tmp/demo_dataset \
  --format=parquet \
  --reader=hf_datasets \
  --batch-size=64 \
  --num-workers=4 \
  --max-batches=100
```

---

## ☸️ Step-by-Step GKE Deployment

For full cloud benchmarking over **GCSFuse sidecar mounts**, **native `gcsfs`**, or **Managed Lustre**, refer to the detailed guide:
👉 [Dataset Loading Step-by-Step Guide](./step_by_step_guide.md)
