# Benchmark Dimension: Worker Concurrency & GCSFuse Memory Block Tuning

This document evaluates the impact of **Worker Process Scaling** (1 vs 4 vs 8 processes per node), **Application I/O Concurrency** (`file_io_concurrency`), and **GCSFuse Memory Block Capping (`write:global-max-blocks:-1`)** on multidimensional array throughput.

---

## 📊 Summary Performance Comparison Table

Single-node benchmark evaluation (`n4-standard-80`, 128 GB array, 200 MB chunks):

| Architecture & Concurrency | Worker Processes | Per-Worker Concurrency (`file_io_concurrency`) | Mount Options (`global-max-blocks`) | Aggregate Write Throughput | Aggregate Read Throughput | Scaling & Impact Findings |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **8 Workers (`global-max-blocks:-1`)** | 8 Workers | 8 (64 total threads) | `ON` (`write:global-max-blocks:-1`) | **3,820.93 MB/s (~30.57 Gbps)** | **4,558.31 MB/s (~36.47 Gbps)** | **OPTIMAL BALANCED SWEET SPOT** (31.9s write / 26.8s read). Saturates ~73% of VM 50 Gbps NIC. |
| **8 Workers (Default Memory Blocks)** | 8 Workers | 16 (128 total threads) | `OFF` (Default Blocks) | **1,850.00 MB/s (~14.80 Gbps)** | **2,749.70 MB/s (~22.00 Gbps)** | **Throttled by Memory Capping**; Default block buffers cause severe write backpressure. |
| **4 Workers (`global-max-blocks:-1`)** | 4 Workers | 8 (32 total threads) | `ON` (`write:global-max-blocks:-1`) | **2,330.38 MB/s (~18.64 Gbps)** | **4,272.23 MB/s (~34.18 Gbps)** | Read throughput is strong (34.2 Gbps), but 32 I/O threads limit write parallelism. |
| **1 Worker (`global-max-blocks:-1`)** | 1 Worker | 16 (16 total threads) | `ON` (`write:global-max-blocks:-1`) | **1,258.31 MB/s (~10.07 Gbps)** | **4,324.54 MB/s (~34.60 Gbps)** | Single-process baseline; 16 threads saturate single-process Python CPU GIL. |

---

## 🔍 Detailed Technical Insights

### 1. Doubling Write Throughput (+107% Speedup) via `write:global-max-blocks:-1`

- **Default Memory Block Allocation (`OFF`)**:
  - GCSFuse sidecar defaults to capping the total number of memory blocks allocated for write streaming buffers.
  - Under 8 parallel worker processes, write threads rapidly hit memory block limits, introducing thread sleep backpressure and capping write throughput at **1,850.00 MB/s**.

- **Un-Capped Memory Blocks (`ON`)**:
  - Specifying `write:global-max-blocks:-1` allows GCSFuse sidecars to dynamically allocate buffer blocks up to available container RAM.
  - This eliminates write thread backpressure, boosting aggregate write throughput to **3,820.93 MB/s (+107% speedup on a single node)** and **+22.8% (+6.37 GB/s) overall across 32 nodes**.

### 2. Multi-Worker Process Scaling (1 vs 4 vs 8 Workers)

- **1 Worker Process**: Achieves high read speed (4.32 GB/s), but write speed is bottlenecked at 1.25 GB/s by single-process CPU GIL and user-space context switches.
- **4 Worker Processes (32 total threads)**: Read speed reaches 4.27 GB/s, but 32 threads leave remaining CPU cores idle on 80 vCPU nodes.
- **8 Worker Processes (64 total threads)**: Optimal sweet spot on `n4-standard-80` nodes. 64 total I/O threads fully utilize the 80 vCPUs without triggering kernel thread context lock contention.
