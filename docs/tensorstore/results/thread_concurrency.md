# Benchmark Dimension: Thread Concurrency & Application I/O Parallelism

This document evaluates the impact of **Application I/O Thread Concurrency** (`file_io_concurrency` per worker process) and total node thread pool scaling on single-node multidimensional array throughput over GCSFuse.

---

## 📊 Summary Performance Comparison Table

Single-node benchmark evaluation (`n4-standard-80`, 128 GB array, 200 MB chunks, 8 worker processes, `write:global-max-blocks:-1` enabled):

| Thread Concurrency Profile | Per-Worker Concurrency (`file_io_concurrency`) | Total Node I/O Threads | Aggregate Write Throughput | Aggregate Read Throughput | Thread Dynamics & Observations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **64 Total Threads (Optimal)** | 8 Threads / Worker | **64 Threads** | **3,820.93 MB/s (~30.57 Gbps)** | **4,558.31 MB/s (~36.47 Gbps)** | **OPTIMAL THREAD POOL BALANCING**; Fully utilizes 80 vCPUs with minimal OS context lock contention. |
| **128 Total Threads (Oversubscribed)** | 16 Threads / Worker | **128 Threads** | **1,850.00 MB/s (~14.80 Gbps)** | **2,749.70 MB/s (~22.00 Gbps)** | **Severe Thread Lock Contention**; High thread oversubscription causes context switching locks and IPC queue stalls. |
| **32 Total Threads** | 8 Threads / 4 Workers | **32 Threads** | **2,330.38 MB/s (~18.64 Gbps)** | **4,272.23 MB/s (~34.18 Gbps)** | Under-utilizes available vCPU hardware threads during parallel write chunk dispatch. |

---

## 🔍 Technical Analysis & Thread Scaling Dynamics

### 1. The Optimal 64-Thread Sweet Spot
- Configuring **8 I/O threads per worker process** across 8 workers yields **64 total node I/O threads**.
- On `n4-standard-80` nodes (80 vCPUs), 64 threads allow high async I/O dispatch to the GCSFuse CSI driver sidecar while leaving 16 vCPUs dedicated to the GCSFuse Go daemon background routines and kernel network stack interrupts.

### 2. Thread Oversubscription & Lock Contention (128 Threads)
- Increasing per-worker concurrency to 16 threads (128 total node threads) exceeds the 80 physical vCPUs.
- Excessive worker thread oversubscription triggers OS kernel thread scheduling latency, lock contention in memory block allocation, and backpressure in GCSFuse FUSE request queues, dropping write throughput by **-51.6%** (to **1,850.00 MB/s**).

---

## 💡 Recommendations

* **Optimal Thread Sizing**: Set `file_io_concurrency` to **8 threads per worker process**.
* **Cap Total Node Threads**: Ensure total I/O threads across all worker processes do not exceed **80% of available node vCPUs** to prevent GCSFuse sidecar daemon starvation.
