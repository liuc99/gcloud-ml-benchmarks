# Hugging Face Parquet Streaming: With vs. Without `manifest.json` Benchmark Report

Empirical benchmark evaluation comparing the impact of **Explicit Metadata Manifest (`manifest.json`)** versus **Runtime Directory Globbing (`*.parquet`)** under **Independent Pod Lifecycle Isolation** (clean GCSFuse sidecar teardown and cold-cache recreation) evaluating **Two-Stage Shuffle** and **Sequential** streaming on GKE with GCSFuse CSI driver.

---

## 🎯 1. Benchmark Objective & Evaluation Scope

Evaluate how metadata discovery and pre-indexing impact startup latency and streaming throughput under clean, isolated Pod lifecycles (destroying GCSFuse daemon and in-memory caches between runs):
- **Target Workload & Scale**: PyTorch multi-worker data loader streaming 10,000 batches (640,000 samples @ `batch_size=64`, ~3.7 GB ingested) across a 1,650-shard Parquet dataset (451.08 GB total volume in GCS).
- **Comparison Matrix**:
  1. **Two-Stage Shuffle Scenario (Independent Isolated Pods)**: 1,650-shard permutation + 10,000-sample in-memory streaming buffer (`buffer_size=10000`, `seed=42`).
  2. **Cache Isolation Protocol**: Completely destroying the Helm release and GCSFuse sidecar daemon between tests to ensure 100% clean, non-polluted cache state.
  3. **Storage Access Mode**: GCSFuse CSI POSIX filesystem mount (`accessMode=gcsfuse`).
- **Key Metrics Tracked**: TTFB, DataLoader preparation latency, sustained read throughput (MB/s), sample ingestion rate (samples/sec), and batch latency percentiles (p50, p95, p99).

---

## ⚙️ 2. Testbed Configuration & Workload Dimensions

| Category | Parameter | Specification / Value |
| :--- | :--- | :--- |
| **Compute & Cluster** | **GKE Environment** | Standard GKE Node Pool (`n4-standard-80`, 80 vCPU, 314.68 GiB RAM) |
| | **Network & MTU** | gVNIC with **8896 Jumbo Frames** |
| **Storage & CSI** | **Storage Backend** | **Google Cloud Storage (GCS) RAPID Zonal** (`gs://chongliu-macrobench-dataset-965f0fed`) |
| | **Storage Access Mode** | **GCSFuse CSI Driver Mount** (`accessMode=gcsfuse`) |
| | **GCSFuse Mount Options** | `implicit-dirs,file-cache:max-size-mb:-1,file-cache:cache-file-for-range-read:true` |
| | **GCSFuse CSI Version** | `v1.22.21-gke.2` |
| **Cache Lifecycle** | **Isolation Strategy** | **Independent Pod Lifecycle Isolation** (Clean Sidecar Teardown per Run) |
| **Model & Dataset** | **Dataset Total Scale** | 1,650 Parquet Shards (451.08 GB total volume in GCS) |
| | **Evaluation Scope** | 10,000 Batches @ `batch_size=64` (640,000 samples, ~3.7 GB ingested per run) |
| | **DataLoader Concurrency** | `num_workers=4`, `prefetch_factor=2` |
| | **Active Shuffle Strategy** | **`two_stage`** (`buffer_size=10000`, `seed=42`) |

---

## 📊 3. Empirical Performance Results (Independent Pod Isolation)

### 1. Two-Stage Shuffle Benchmark Comparison (Clean Pod Lifecycle Isolation)

| Benchmark Evaluation Metric | Baseline: Without Manifest (Dynamic Glob on Clean Pod) | Optimized: With Manifest (Direct Read from Bucket `manifest.json`) | Performance Delta & Impact |
| :--- | :---: | :---: | :--- |
| **Dataset Shard Scale** | 1,650 Shards (451.08 GB) | 1,650 Shards (451.08 GB) | 100% Strict Parity |
| **Time to First Batch (TTFB)** | **1,999.02 ms (1.999 s)** | **1,839.81 ms (1.840 s)** | ⚡ **-159.21 ms Faster (-8.0% Latency)** |
| **Read Throughput** | **83.09 MB/s (0.65 Gbps)** | **83.93 MB/s (0.66 Gbps)** | 🚀 **+0.84 MB/s (+1.01% Higher)** |
| **Sample Ingestion Speed** | **14,118.00 samples/s** | **14,260.81 samples/s** | 🚀 **+142.81 samples/s Faster** |
| **10,000 Batch Total Ingestion Duration** | **45.3322 s** | **44.8782 s** | ⏱️ **0.454s faster overall** |
| **Batch Latency (p50 / p95)** | 1.90 ms / 4.87 ms | 1.91 ms / 4.90 ms | Steady-state parity (~1.9ms) |
| **Batch Latency (p99)** | 8.52 ms | 8.85 ms | Steady-state parity (~8.6ms) |

---

## 🔬 4. Technical Analysis & Key Findings

### 1. Hugging Face Datasets Multi-Worker Concurrency Throttling Trap
When streaming datasets from Cloud Storage, a critical trap exists in Hugging Face `datasets.load_dataset(..., streaming=True)`:
- **Passing a single glob string (`data_files="*.parquet"`)**: Hugging Face internally assigns `dataset.n_shards = 1`. When PyTorch `DataLoader` initializes `num_workers=4`, it detects that `num_workers > num_shards`, logs `Too many dataloader workers: 4 (max is dataset.num_shards=1). Stopping 3 dataloader workers.`, and **kills 3 workers, throttling ingestion to a single thread and prematurely aborting once shard 0 finishes**.
- **Passing an explicit shard list (`data_files=shards`)**: Hugging Face sets `dataset.n_shards = 1650`, enabling all 4 workers to stream concurrently across all 10,000 batches.
- **Why Manifest is Essential**: Without `manifest.json`, the user application is forced to run `glob.glob` on the FUSE mount at container startup to discover the list of shards. With `manifest.json`, the client immediately loads the pre-indexed shard list in memory in `<1ms`.

### 2. GCSFuse Cache Teardown Verification
- Under the **Independent Pod Lifecycle Isolation Protocol**, tearing down the Helm release causes the `gke-gcsfuse-sidecar` container and its FUSE daemon to terminate completely, freeing all userspace Stat Cache and Type Cache.
- In both clean Pod runs, once all 1,650 shards are provided to the DataLoader, steady-state Two-Stage Shuffle throughput stabilizes at **~83–84 MB/s**, with the manifest providing an **8.0% TTFB startup reduction** and eliminating runtime filesystem scanning.

---

## 💡 5. Production Recommendations

1. **Always Supply Explicit Shard Lists or `manifest.json`**:
   - Never pass raw glob strings to Hugging Face streaming datasets in production multi-worker DataLoader environments to prevent worker throttling and premature epoch completion.
2. **Maintain Bucket-Level `manifest.json`**:
   - Pre-generating `manifest.json` directly inside the GCS bucket (`gs://<bucket>/manifest.json`) standardizes metadata access across MaxText, PyTorch, and Hugging Face loaders.
