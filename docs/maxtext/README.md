# MaxText Dataset Ingestion & GCS Range Read / ArrayRecord Benchmark

A dedicated workload and demo suite within `gcloud-ml-benchmarks` simulating the **MaxText JAX LLM Training Input Pipeline** reading multi-column Parquet datasets via **GCS Range Reads** and pre-tokenized **ArrayRecord** format.

---

## 🎯 Architecture & Format Overview

In large-scale LLM training with MaxText (e.g. Llama 3 / Gemma pre-training), datasets are typically ingested in one of two primary formats:

### 1. Parquet Format (On-the-fly Tokenization + GCS Range Reads)
- **GCS Range Reads**: MaxText input pipelines use GCS Range Requests (`Range: bytes=start-end`) to fetch Parquet footers and project target columns (`input_ids`, `label`), bypassing unneeded metadata and saving 50%+ network bandwidth.
- **Flexibility**: Enables instant training start and on-the-fly tokenization/data augmentation without pre-processing delays.

### 2. ArrayRecord Format (Pre-tokenized Zero-CPU Streaming)
- **Pre-tokenized Int32 Arrays**: Raw text in Parquet is pre-processed into `.array_record` shards holding pre-tokenized `int32` token arrays via the included [`parquet_to_arrayrecord.py`](../../workloads/maxtext-parquet-loader/helm_chart/parquet_to_arrayrecord.py) converter.
- **Zero-CPU Overhead**: Eliminates runtime tokenizer latency and CPU decoding bottlenecks during training steps.
- **Sub-millisecond Random Access**: ArrayRecord footer index tables enable $O(1)$ random sample indexing with sub-millisecond batch latencies (~0.33 ms/batch).

---

## 📊 Benchmark Comparison: Parquet vs. ArrayRecord & Shuffle Modes

| Format & Strategy | Time to First Batch (TTFB) | Upfront Index Scanning Penalty | Batch Latency (p50 / p95) | Main Characteristic / Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **Parquet (`none` / `two_stage`)** | **~372 ms** | 0 ms | ~1.5 ms / 4.2 ms | Instant start; zero pre-processing waiting time. |
| **Parquet (`global` shuffle)** | 91.64 s | **91.47 s** | ~0.01 ms | ⚠️ High upfront index scanning penalty over 1600+ Parquet footers. |
| **ArrayRecord (`none` / `two_stage`)**| ~4.7 s – 7.0 s | 0 ms | **0.34 ms / 0.57 ms** | Sub-millisecond batch latency, zero runtime CPU tokenization. |
| **ArrayRecord (`global` shuffle)** | ~6.5 s | **31.56 ms** | **0.33 ms / 0.48 ms** | **🚀 2900x faster index loading** than Parquet global shuffle with true randomness. |

---

## 🔌 Dual Access Modes Supported

| Access Mode | URI Path Format | Driver / Interface | Range Read / Streaming Mechanism |
| :--- | :--- | :--- | :--- |
| **1. Native GCS Client** | `gs://my-bucket/dataset` | `pyarrow.fs.GCSFileSystem` / `gcsfs` / `google-cloud-storage` | Direct HTTP GET Range Requests (`Range: bytes=X-Y`) or gRPC stream |
| **2. GCSFuse Sidecar Mount** | `/gcs/my-bucket/dataset` | GCSFuse CSI Driver (`GcsFuseCsiDriver`) POSIX `lseek` + `read` | FUSE kernel layer translates POSIX seeks into GCS Range Requests |

---

## 📖 Step-by-Step Guide & Demos

For step-by-step instructions on running benchmarks and conversion scripts on GKE (or locally), refer to:
👉 [MaxText Parquet & ArrayRecord Benchmark Guide](./parquet_range_reads_guide.md)
