# ML Dataset Loading Benchmarks & PoC Suite

A dedicated benchmarking and PoC framework within `gcloud-ml-benchmarks` for evaluating high-throughput ML dataset streaming, batch ingestion speed, and sample access latency on Google Cloud Platform (GCP) and Google Kubernetes Engine (GKE).

---

## 🚀 Key Features

1. **Synthetic Dataset Generator (`dataset_generator.py`)**:
   - Generates benchmark datasets directly on GCS (`gs://...`), GCSFuse mounts (`/gcs/...`), Google Cloud Managed Lustre (`/lustre/...`), or local disk.
   - Supported Formats: Parquet, WebDataset TAR, Zarr / TensorStore, PyTorch `.pt`, and JSONL.

2. **Dataset Loading Benchmark Harness (`dataset_loading_bench.py`)**:
   - Evaluates key data loader frameworks: HuggingFace `datasets`, WebDataset, TensorStore, PyTorch `DataLoader`, and Native PyArrow / GCSFS.
   - Collects core performance metrics: Time to First Batch (TTFB), Read Throughput (MB/s and Gbps), Ingestion Speed (samples/sec), and Batch Latency Percentiles (p50, p95, p99 ms).

3. **Helm & Kubernetes JobSet Integration**:
   - Deploy single-node and multi-node dataset loading benchmark releases on GKE using Helm and Kubernetes JobSet.

---

## 📊 Benchmark Test Results

- [ArrayRecord Range Read Chunk Size Scaling on GCSFuse CSI Driver](results/arrayrecord_range_read_chunk_size_scaling.md)
  - **Dimension**: Chunked Range Read scaling (8 KB, 64 KB, 256 KB, 512 KB) under Zero-Buffer True Global Shuffle on GCSFuse CSI (1,650 shards, 134.3M records).
  - **Highlights**: TTFB invariance verification, 2.05x throughput speedup on GCSFuse (2,812 -> 5,764 samples/s / 45.0 MB/s), and p99 latency reduction.
- [Hugging Face Parquet Manifest (`manifest.json`) vs. Glob Comparison](results/hf_parquet_manifest_comparison.md)
  - **Dimension**: Explicit JSON Shard Manifest vs. Dynamic Runtime Globbing under Independent Pod Lifecycle Isolation.
  - **Highlights**: Multi-worker concurrency throttling traps, metadata storm elimination, and startup TTFB improvements.
- [GCSFuse vs. Direct GCS (`gcsfs`) Parquet Streaming Comparison](results/gcsfuse_vs_direct_gcs_parquet.md)
  - **Dimension**: GCSFuse CSI Driver Mount vs. Native GCSFS Python Client.
  - **Highlights**: TTFB cold start latency (200ms vs 2.5s), read throughput (137 MB/s vs 27 MB/s), and PyTorch DataLoader multiprocessing fork-safety.
- [Multi-Format & Storage Backend Performance Comparison](results/format_comparison.md)
  - **Dimension**: Parquet vs ArrayRecord vs WebDataset vs Zarr vs PyTorch `.pt`.
  - **Highlights**: Ingestion throughput, host CPU utilization, random seek efficiency, and storage cache suitability across backends.

---

## 📖 Step-by-Step Guides & Deployment

- [Dataset Loading Step-by-Step Guide](./step_by_step_guide.md): Complete instructions for running benchmarks across GCSFuse, native `gcsfs`, and Managed Lustre.
- [Workload Quickstart & Helm Reference](../../workloads/multi-format-dataset-loader/README.md): Operator guide for Helm parameters and CLI tools.
