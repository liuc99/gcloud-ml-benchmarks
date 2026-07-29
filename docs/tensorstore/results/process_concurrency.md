# Benchmark Dimension: Worker Process Concurrency

This document evaluates the impact of **Worker Process Scaling** (comparing **1 Worker**, **4 Workers**, and **8 Workers** per node) on single-node multidimensional array read and write throughput over GCSFuse.

---

## 📊 Summary Performance Comparison Table

Single-node benchmark evaluation (`n4-standard-80`, 128 GB array, 200 MB chunks, `write:global-max-blocks:-1` enabled):

| Worker Process Count | Total Node Process Topology | Aggregate Write Throughput | Aggregate Read Throughput | Scaling Efficiency & Observations |
| :--- | :--- | :--- | :--- | :--- |
| **8 Workers (Optimal)** | 8 Worker Processes / Node | **3,820.93 MB/s (~30.57 Gbps)** | **4,558.31 MB/s (~36.47 Gbps)** | **OPTIMAL MULTI-PROCESS SWEET SPOT**; Distributes memory serialization across 8 process boundaries, bypassing GIL limits. |
| **4 Workers** | 4 Worker Processes / Node | **2,330.38 MB/s (~18.64 Gbps)** | **4,272.23 MB/s (~34.18 Gbps)** | Strong read performance (34.2 Gbps), but 4 processes leave write CPU cores under-utilized on 80 vCPU nodes. |
| **1 Worker** | 1 Worker Process / Node | **1,258.31 MB/s (~10.07 Gbps)** | **4,324.54 MB/s (~34.60 Gbps)** | **Single-Process GIL Bottleneck**; Python CPU GIL and single-process context switches lock write serialization. |

---

## 🔍 Technical Analysis & Multi-Process Scaling Behavior

### 1. The Single-Process GIL Bottleneck (1 Worker Process)
- **Read Behavior**: Single-worker reads achieve strong throughput (**4.32 GB/s**) because read buffer transfers bypass Python GIL locks via direct zero-copy C++ TensorStore bindings.
- **Write Behavior**: Single-worker write speed is severely bottlenecked at **1,258.31 MB/s**. Python memory layout serialization and single-process user-space context switches lock the interpreter execution thread.

### 2. Multi-Worker Process Scaling (4 vs 8 Workers)
- **4 Worker Processes**: Increases write throughput to **2,330.38 MB/s (+85% vs 1 worker process)**, but leaves half the remaining CPU cores idle on 80 vCPU (`n4-standard-80`) VM nodes.
- **8 Worker Processes**: Achieves peak write speed (**3,820.93 MB/s**) and peak read speed (**4,558.31 MB/s**). Distributing I/O across 8 worker processes effectively circumvents single-process GIL contention while matching the 80 vCPU core topology.

---

## 💡 Recommendations

* **Process Count Matching**: On 80 vCPU nodes (`n4-standard-80`), launch **8 worker processes** (e.g., 8 PyTorch DDP ranks or 8 Python worker processes).
* **Multi-Process Partitioning**: Partition array chunk slices evenly across independent worker process instances to maximize parallel CPU buffer serialization.
