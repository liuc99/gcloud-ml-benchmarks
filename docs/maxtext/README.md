# MaxText Dataset Ingestion & Storage Benchmark Suite

A dedicated workload and demo suite within `gcloud-ml-benchmarks` simulating the **MaxText JAX LLM Training Input Pipeline** reading multi-column Parquet datasets via **GCS Range Reads** and pre-tokenized **ArrayRecord** format.

---

## 🏗️ Architecture & Format Overview

In large-scale LLM training with MaxText (e.g. Llama 3 / Gemma pre-training), datasets are typically ingested in one of two primary formats:

### 1. Parquet Format (On-the-fly Tokenization + GCS Range Reads)
- **GCS Range Reads**: MaxText input pipelines use GCS Range Requests (`Range: bytes=start-end`) to fetch Parquet footers and project target columns (`input_ids`, `label`), bypassing unneeded metadata and saving 50%+ network bandwidth.
- **Flexibility**: Enables instant training start and on-the-fly tokenization/data augmentation without pre-processing delays.

### 2. ArrayRecord Format (Pre-tokenized Zero-CPU Streaming)
- **Pre-tokenized Int32 Arrays**: Raw text in Parquet is pre-processed into `.array_record` shards holding pre-tokenized `int32` token arrays via the included multi-process converter.
- **Zero-CPU Overhead**: Eliminates runtime tokenizer latency and CPU decoding bottlenecks during training steps.
- **Sub-millisecond Random Access**: ArrayRecord footer index tables enable $O(1)$ random sample indexing with sub-millisecond batch latencies (~0.33 ms/batch).

### 3. In-Tree Standalone DataLoader (`loaderMode=in_tree_loader`)
- **Native Pipeline Evaluation**: Runs MaxText's native Grain / TFDS data pipeline with `jax.block_until_ready()`.
- **End-to-End Throughput**: Measures actual training data loading throughput, TTFB, and per-step batch latency on CPU/TPU without model backward pass computation.

---

## 📊 Benchmark Test Results

The empirical results from cloud testing on GKE (`n4-standard-80`, MTU 8896, GCSFuse CSI `v1.22.21-gke.1`) are organized into modular reports:

1. [Parquet vs. ArrayRecord Performance](results/parquet_vs_arrayrecord.md)
   - **Dimension**: Storage footprint (420 GB vs 155 GB), CPU overhead, TTFB, and step latency.
   - **Highlights**: 63.9% storage and network bandwidth reduction, zero-CPU tokenization overhead, +48.7% effective tensor throughput.

2. [Shuffle Strategies: None vs. Two-Stage vs. Global](results/shuffle_strategies.md)
   - **Dimension**: Shuffling algorithms, startup cold-start delays, and index scanning penalties.
   - **Highlights**: Parquet global shuffle incurs 91.47s metadata scan penalty; ArrayRecord index loading completes in 31.56ms (2900x faster).

3. [Storage Access Modes: GCSFuse CSI vs. Native GCS](results/storage_access_modes.md)
   - **Dimension**: `/gcs/...` POSIX mount vs direct `gs://...` client streaming.
   - **Highlights**: Mount option recommendations (`stat-cache-capacity`, `file-cache`) and architecture selection guide.

4. [ArrayRecord Range Read Chunk Size Scaling on GCSFuse CSI Driver](../multi_format_dataset/results/arrayrecord_range_read_chunk_size_scaling.md)
   - **Dimension**: Chunked Range Read scaling (8 KB, 64 KB, 256 KB, 512 KB) under Zero-Buffer True Global Shuffle on GCSFuse.
   - **Highlights**: TTFB invariance verification across chunk sizes, +105.0% throughput gain on GCSFuse (2,812 -> 5,764 samples/s), and p99 latency reduction.

---

## 📖 Step-by-Step Guides & Deployment

- [MaxText Parquet & ArrayRecord Benchmark Guide](./parquet_range_reads_guide.md): Complete instructions for generating datasets, converting formats, and deploying on GKE.
- [Workload Quickstart & Helm Reference](../../workloads/maxtext-dataset-loader/README.md): Operator guide for Helm parameters and CLI tools.
