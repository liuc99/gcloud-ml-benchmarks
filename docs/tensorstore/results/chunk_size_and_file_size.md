# Benchmark Dimension: Zarr Chunk Size & Sub-Second Slice Retrieval

This document evaluates the impact of **Zarr Chunk Size** (comparing **50 MB**, **200 MB**, and **400 MB** chunks) and array slicing granularity on read/write throughput and metadata latency when accessing TensorStore multidimensional arrays over GCSFuse.

---

## 📊 Summary Benchmark Table

Single-node benchmark evaluation (`n4-standard-80`, 128 GB total dataset size `(16000, 8000, 250)` float32 elements):

| Chunk Size | Total Chunks / Files | Single-Worker Write Throughput | Single-Worker Read Throughput | Multi-Worker Peak Write (8 Workers) | Multi-Worker Peak Read (8 Workers) | Sub-Second Slice Latency | Performance Impact & Observations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **400 MB** | 320 Files | **1,483.53 MB/s (~11.87 Gbps)** | **3,853.03 MB/s (~30.82 Gbps)** | 3,610.12 MB/s | 4,410.50 MB/s | 0.512s | **PEAK SINGLE-WORKER WRITE SPEED (+18% vs 200MB)**; Reduces per-file VFS metadata overhead. |
| **200 MB** | 640 Files | **1,258.31 MB/s (~10.07 Gbps)** | **4,324.54 MB/s (~34.60 Gbps)** | **3,820.93 MB/s (~30.57 Gbps)** | **4,558.31 MB/s (~36.47 Gbps)** | **0.3376s** | **OPTIMAL BALANCED SWEET SPOT**; Peak read speed + sub-second (0.33s) slice retrieval. |
| **50 MB** | 2,560 Files | **252.29 MB/s (~2.02 Gbps)** | **2,210.82 MB/s (~17.69 Gbps)** | 1,120.45 MB/s | 2,980.10 MB/s | 0.285s | **SEVERELY DEGRADED (-80% Write / -49% Read Penalty)**; Massive 2,560 file open/create VFS overhead. |

---

## 🔍 Detailed Analysis & Observations

1. **The Metadata Overhead Bottleneck (50 MB Chunks)**:
   - Writing 128 GB as 2,560 individual 50 MB files incurs significant VFS kernel context switching and GCS object creation metadata roundtrips.
   - Single-worker write throughput drops to **252.29 MB/s** (an 80% drop compared to 200 MB chunks).

2. **200 MB Chunk Size Sweet Spot**:
   - 200 MB chunks strike an ideal balance: small enough to enable fine-grained sub-second multi-dimensional slicing (**0.3376s** to extract a 200 MB array slice out of a 128 GB dataset), yet large enough to allow GCSFuse and TCP streaming buffers to reach full 36+ Gbps line rate.

3. **400 MB Chunks for Write-Heavy Operations**:
   - Single-worker write speed peaks at **1,483.53 MB/s** (+18% faster than 200 MB chunks) due to fewer file creation metadata roundtrips.
