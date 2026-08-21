# TensorStore Worker Process Concurrency Benchmark Report

Empirical benchmark evaluation measuring the throughput and scaling efficiency of **1, 4, and 8 Worker Processes per Node** during TensorStore multidimensional array operations over GCSFuse on GKE.

---

## 🎯 1. Benchmark Objective & Evaluation Scope

Evaluate multi-process scaling, Python GIL serialization limits, and core utilization:
- **Target Workload & Scale**: Single-node TensorStore Zarr array (128 GB volume, 200 MB chunk sizing) on an `n4-standard-80` instance.
- **Comparison Matrix**: **1 Worker Process** vs. **4 Worker Processes** vs. **8 Worker Processes**.
- **Key Metrics Tracked**: Aggregate write throughput (MB/s), aggregate read throughput (MB/s), and CPU core scaling efficiency.

---

## ⚙️ 2. Testbed Configuration & Workload Dimensions

| Category | Parameter | Specification / Value |
| :--- | :--- | :--- |
| **Compute & Cluster** | **GKE Environment** | Standard GKE Node Pool (`n4-standard-80`, 80 vCPU, 314 GiB RAM) |
| | **Network & MTU** | gVNIC with **8896 Jumbo Frames** |
| **Storage & CSI** | **Storage Backend** | **Google Cloud Storage (GCS) RAPID Zonal** |
| | **GCSFuse CSI Version** | `v1.22.21-gke.1` (`global-max-blocks:-1`) |
| **Model & Checkpoint** | **Array Volume** | 128 GB Multidimensional Float32 Array (200 MB chunks) |
| **Testing Methodology** | **Repetition & Aggregation** | 3 consecutive runs per configuration (Median reported) |

---

## 📊 3. Empirical Performance Results & Comparison

| Worker Process Count | Total Node Process Topology | Aggregate Write Throughput | Aggregate Read Throughput | Scaling Efficiency & Observations |
| :--- | :--- | :--- | :--- | :--- |
| **8 Workers (Optimal)** | 8 Worker Processes / Node | **3,820.93 MB/s (~30.57 Gbps)** | **4,558.31 MB/s (~36.47 Gbps)** | **Optimal Multi-Process Sweet Spot**; Distributes serialization across 8 processes. |
| **4 Workers** | 4 Worker Processes / Node | **2,330.38 MB/s (~18.64 Gbps)** | **4,272.23 MB/s (~34.18 Gbps)** | Strong read speed, but leaves write CPU cores under-utilized on 80 vCPU nodes. |
| **1 Worker** | 1 Worker Process / Node | **1,258.31 MB/s (~10.07 Gbps)** | **4,324.54 MB/s (~34.60 Gbps)** | **Single-Process GIL Bottleneck**; Python CPU serialization locks write speed. |

### Key Findings
1. **3.0x Write Speedup with 8 Workers**: Distributing array writing across 8 worker processes pushed write throughput from **1.26 GB/s to 3.82 GB/s**.
2. **C++ Read Independence**: Reads remain high (>4.2 GB/s) across all configurations because C++ TensorStore zero-copy buffers bypass Python GIL locks.

---

## 🔬 4. Technical Analysis & Deep-Dive Insights

### 1. The Write GIL Bottleneck
In single-process configurations, memory buffer allocation and chunk serialization lock the Python interpreter GIL, capping write throughput at ~1.26 GB/s.

### 2. Multi-Process CPU Parallelism
Launching 8 independent processes allocates dedicated memory spaces and native thread pools, saturating CPU serialization pipelines and delivering **3.82 GB/s write / 4.56 GB/s read**.

---

## 💡 5. Production Recommendations & Related Documentation

### 1. Process Concurrency Guidelines
- **Target Concurrency**: On 80 vCPU nodes (`n4-standard-80`), configure **8 worker processes** (e.g. 8 DDP ranks or multi-process dataloaders).

### 2. Related Documentation
- [Thread Concurrency Impact](./thread_concurrency.md)
- [Chunk Sizing Evaluation](./chunk_size_and_file_size.md)
- [TensorStore Overview](../README.md)
