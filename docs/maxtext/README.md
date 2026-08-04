# MaxText Parquet Dataset Ingestion & GCS Range Read Benchmark

A dedicated workload and demo suite within `gcloud-ml-benchmarks` simulating the **MaxText JAX LLM Training Input Pipeline** reading multi-column Parquet datasets via **GCS Range Reads**.

---

## 🎯 Architecture Overview

In large-scale LLM training with MaxText (e.g. Llama 3 / Gemma pre-training), datasets are stored in Parquet format containing multiple columns (such as `input_ids`, `attention_mask`, `labels`, and optional `metadata_bytes`).

Instead of downloading entire multi-gigabyte Parquet files, MaxText input pipelines use **GCS Range Reads** (`Range: bytes=start-end`) to:
1. **Fetch Parquet Footer Metadata**: Read the final 64 KB – 1 MB footer to parse row group offsets and column chunk boundaries.
2. **Project Target Columns**: Issue targeted Range Requests to download *only* the specific token columns (`input_ids`, `labels`) required for model steps, bypassing unneeded columns and saving 50%+ network bandwidth and memory.

---

## 🔌 Dual Access Modes Supported

This harness supports both primary GCS dataset access methods on GCP / GKE:

| Access Mode | URI Path Format | Driver / Interface | Range Read Mechanism | Recommended Range Read Options |
| :--- | :--- | :--- | :--- | :--- |
| **1. Native GCS Client** | `gs://my-bucket/parquet` | `pyarrow.fs.GCSFileSystem` / `gcsfs` / `google-cloud-storage` | Direct HTTP GET Range Requests (`Range: bytes=X-Y`) or gRPC Byte Ranges | Default native GCS connection pool |
| **2. GCSFuse Sidecar Mount** | `/gcs/my-bucket/parquet` | GCSFuse CSI Driver (`GcsFuseCsiDriver`) POSIX `lseek` + `read` | FUSE kernel layer translates POSIX seeks into GCS Range Requests | `file-cache:cache-file-for-range-read:true`, `file-cache:max-size-mb:-1` |

---

## 📊 Core Performance & Range Read Metrics Collected

- ⏱️ **Time to First Batch (TTFB)**: Delay from script start to first batch ready for JAX step (ms).
- 📑 **Parquet Footer Parse Latency**: Time to retrieve and parse metadata via tail range read (ms).
- 🎯 **Range Read Efficiency (%)**: Useful feature payload bytes vs total bytes downloaded from GCS.
- ⚡ **GCS Range Request Latency**: Average and p95 latency per range read request (ms).
- 🚀 **Ingestion Throughput**: Total read speed (MB/s and Gbps).

---

## 📖 Step-by-Step Guide

For step-by-step execution instructions on GKE (or locally), refer to:
👉 [MaxText Parquet GCS Range Read Guide](./parquet_range_reads_guide.md)
