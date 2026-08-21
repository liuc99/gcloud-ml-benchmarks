# TensorStore Thread Concurrency Benchmark Report

Empirical benchmark evaluation measuring the throughput and thread lock dynamics of **32, 64, and 128 Total Node Threads** during TensorStore multidimensional array operations over GCSFuse on GKE.

---

## 🎯 1. Benchmark Objective & Evaluation Scope

Evaluate worker I/O thread concurrency, thread pool saturation, and kernel scheduler contention:
- **Target Workload & Scale**: Single-node TensorStore array (128 GB volume, 200 MB chunks, 8 worker processes) on an `n4-standard-80` node.
- **Comparison Matrix**: **32 Total Threads** vs. **64 Total Threads (Optimal)** vs. **128 Total Threads (Oversubscribed)**.
- **Key Metrics Tracked**: Aggregate write throughput (MB/s), aggregate read throughput (MB/s), and OS context switching impact.

---

## ⚙️ 2. Testbed Configuration & Workload Dimensions

| Category | Parameter | Specification / Value |
| :--- | :--- | :--- |
| **Compute & Cluster** | **GKE Environment** | Standard GKE Node Pool (`n4-standard-80`, 80 vCPU, 314 GiB RAM) |
| | **Network & MTU** | gVNIC with **8896 Jumbo Frames** |
| **Storage & CSI** | **Storage Backend** | **Google Cloud Storage (GCS) RAPID Zonal** |
| | **GCSFuse CSI Version** | `v1.22.21-gke.1` (`global-max-blocks:-1`) |
| **Model & Checkpoint** | **Thread Profiles** | 4 to 16 threads per worker process across 8 processes |
| **Testing Methodology** | **Repetition & Aggregation** | 3 consecutive runs per configuration (Median reported) |

---

## 📊 3. Empirical Performance Results & Comparison

| Thread Concurrency Profile | Per-Worker Concurrency | Total Node I/O Threads | Aggregate Write Throughput | Aggregate Read Throughput | Thread Dynamics & Observations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **64 Total Threads (Optimal)** | 8 Threads / Worker | **64 Threads** | **3,820.93 MB/s (~30.57 Gbps)** | **4,558.31 MB/s (~36.47 Gbps)** | **Optimal Thread Pool Balancing**; Fully utilizes 80 vCPUs with minimal context lock. |
| **128 Total Threads (Oversubscribed)** | 16 Threads / Worker | **128 Threads** | **1,850.00 MB/s (~14.80 Gbps)** | **2,749.70 MB/s (~22.00 Gbps)** | **Severe Thread Lock Contention**; Oversubscription triggers scheduler thrashing (-51.6%). |
| **32 Total Threads** | 8 Threads / 4 Workers | **32 Threads** | **2,330.38 MB/s (~18.64 Gbps)** | **4,272.23 MB/s (~34.18 Gbps)** | Under-utilizes available vCPU hardware threads during chunk dispatch. |

### Key Findings
1. **64-Thread Sweet Spot**: 64 total I/O threads delivers peak **3.82 GB/s write and 4.56 GB/s read** by matching available vCPU resources while leaving 16 vCPUs for background daemon routines.
2. **-51.6% Drop Under 128 Threads**: Exceeding the 80 physical vCPUs triggers thread lock contention and scheduler overhead, cutting write performance in half.

---

## 🔬 4. Technical Analysis & Deep-Dive Insights

### 1. The 64-Thread Balanced Headroom
8 threads per process $\times$ 8 workers = 64 worker threads. This leaves 16 cores free on the 80-vCPU instance for the GCSFuse Go daemon background sync routines and network softirqs.

### 2. Thread Oversubscription Thrashing
At 128 threads, OS context switches and mutex contention on memory buffer pools create write backpressure, degrading I/O queue throughput.

---

## 💡 5. Production Recommendations & Related Documentation

### 1. Thread Pool Best Practices
- **Set `file_io_concurrency=8`**: On 80 vCPU nodes, restrict per-worker concurrency to 8 threads.
- **Cap Total Node Threads at 80% vCPUs**: Ensure total application threads do not starve the GCSFuse sidecar daemon.

### 2. Related Documentation
- [Worker Process Concurrency](./process_concurrency.md)
- [Chunk Sizing Evaluation](./chunk_size_and_file_size.md)
- [TensorStore Overview](../README.md)
