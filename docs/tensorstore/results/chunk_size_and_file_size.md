# TensorStore Zarr Chunk Size & Slicing Latency Benchmark Report

Empirical benchmark evaluation measuring the throughput and metadata latency impact of **50 MB, 200 MB, and 400 MB Zarr Chunk Sizes** during TensorStore operations over GCSFuse on GKE.

---

## 🎯 1. Benchmark Objective & Evaluation Scope

Evaluate chunk granularity, file count metadata overhead, and sub-second slice retrieval latency:
- **Target Workload & Scale**: Single-node TensorStore Zarr array (128 GB volume, shape `(16000, 8000, 250)` Float32) on `n4-standard-80`.
- **Comparison Matrix**: **50 MB Chunks** (2,560 files) vs. **200 MB Chunks** (640 files) vs. **400 MB Chunks** (320 files).
- **Key Metrics Tracked**: Single-worker and multi-worker write/read throughput (MB/s), and slice retrieval latency.

---

## ⚙️ 2. Testbed Configuration & Workload Dimensions

| Category | Parameter | Specification / Value |
| :--- | :--- | :--- |
| **Compute & Cluster** | **GKE Environment** | Standard GKE Node Pool (`n4-standard-80`, 80 vCPU, 314 GiB RAM) |
| | **Network & MTU** | gVNIC with **8896 Jumbo Frames** |
| **Storage & CSI** | **Storage Backend** | **Google Cloud Storage (GCS) RAPID Zonal** |
| | **GCSFuse CSI Version** | `v1.22.21-gke.1` (`global-max-blocks:-1`) |
| **Model & Checkpoint** | **Array Volume** | 128 GB Float32 TensorStore Array |
| **Testing Methodology** | **Repetition & Aggregation** | 3 consecutive runs per configuration (Median reported) |

---

## 📊 3. Empirical Performance Results & Comparison

| Chunk Size | Total Files | Single-Worker Write | Single-Worker Read | Peak Write (8 Workers) | Peak Read (8 Workers) | Slice Latency | Performance Impact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **400 MB** | 320 Files | **1,483.53 MB/s (~11.87 Gbps)** | **3,853.03 MB/s (~30.82 Gbps)** | 3,610.12 MB/s | 4,410.50 MB/s | 0.512s | **Peak single-worker write speed (+18%)**; Minimizes file metadata overhead. |
| **200 MB** | 640 Files | **1,258.31 MB/s (~10.07 Gbps)** | **4,324.54 MB/s (~34.60 Gbps)** | **3,820.93 MB/s (~30.57 Gbps)** | **4,558.31 MB/s (~36.47 Gbps)** | **0.3376s** | **Optimal Balanced Sweet Spot**; Peak read speed + sub-second (0.33s) slice retrieval. |
| **50 MB** | 2,560 Files | **252.29 MB/s (~2.02 Gbps)** | **2,210.82 MB/s (~17.69 Gbps)** | 1,120.45 MB/s | 2,980.10 MB/s | 0.285s | **Severe Penalty (-80% Write / -49% Read)**; Excessive 2,560 file open/create overhead. |

### Key Findings
1. **200 MB Sweet Spot**: Balances fast multi-worker throughput (4.56 GB/s read / 3.82 GB/s write) with sub-second (**0.33s**) random slice extraction.
2. **Metadata Storm at 50 MB**: Writing 2,560 files drops single-worker write throughput by **80%** (252 MB/s) due to kernel VFS file creation calls.

---

## 🔬 4. Technical Analysis & Deep-Dive Insights

### 1. The Small File Metadata Storm (50 MB)
Creating thousands of small chunk files incurs serialized VFS inode creation and remote GCS object metadata roundtrips, causing severe I/O backpressure.

### 2. Large Block Streaming (200–400 MB)
Chunks sized 100 MB–400 MB allow GCSFuse and the Linux read-ahead buffer to saturate network sockets while keeping file counts strictly bounded.

---

## 💡 5. Production Recommendations & Related Documentation

### 1. Chunk Sizing Guidelines
- **Balanced ML Training & Checkpointing**: Size chunks between **100 MB and 200 MB** for optimal slice performance and high write throughput.
- **Write-Heavy Ingestion Pipelines**: Use **200 MB to 400 MB** chunks to minimize object creation overhead.

### 2. Related Documentation
- [Worker Process Concurrency](./process_concurrency.md)
- [Thread Concurrency Impact](./thread_concurrency.md)
- [TensorStore Overview](../README.md)
