# Benchmark Dimension: GCSFuse Memory Block Buffer Tuning (`write:global-max-blocks`)

This document evaluates the impact of **GCSFuse Memory Block Allocation Capping (`write:global-max-blocks:-1`)** versus default memory block caps on multidimensional array write throughput and memory buffer utilization.

---

## 📊 Summary Performance Comparison Table

Single-node benchmark evaluation (`n4-standard-80`, 128 GB array, 8 worker processes, 200 MB chunks):

| Mount Option State | `global-max-blocks` Parameter | Aggregate Write Throughput | Aggregate Read Throughput | Write Speedup / Impact | Findings & Memory Dynamics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Un-Capped (`ON`)** | `write:global-max-blocks:-1` | **3,820.93 MB/s (~30.57 Gbps)** | **4,558.31 MB/s (~36.47 Gbps)** | **+107% Write Speedup Baseline** | **OPTIMAL MEMORY STREAMING** (31.9s write). Dynamic block allocation prevents thread stalls. |
| **Default Capped (`OFF`)** | Default Memory Blocks | **1,850.00 MB/s (~14.80 Gbps)** | **2,749.70 MB/s (~22.00 Gbps)** | **Throttled Baseline (-51.6% Penalty)** | **Write Backpressure Bottleneck**; Memory block pool saturation causes worker thread sleep cycles. |

---

## 🔍 Technical Insights & Memory Dynamics

### 1. Doubling Write Throughput (+107% Speedup) via `write:global-max-blocks:-1`

- **Default Memory Block Allocation (`OFF`)**:
  - By default, the GCSFuse sidecar caps the maximum number of memory blocks allocated for write streaming buffers.
  - When 8 parallel worker processes stream large multidimensional Zarr array chunks simultaneously, write buffer queues rapidly exhaust available block tokens.
  - This forces GCSFuse streaming threads into sleep backpressure loops, throttling write throughput at **1,850.00 MB/s**.

- **Un-Capped Memory Blocks (`ON`)**:
  - Specifying `write:global-max-blocks:-1` allows GCSFuse sidecar daemons to dynamically allocate streaming buffer blocks up to total container RAM.
  - This eliminates write thread backpressure entirely, boosting aggregate write throughput to **3,820.93 MB/s (+107% speedup on a single node)** and **+22.8% (+6.37 GB/s) aggregate gain across 32 nodes**.

---

## ⚙️ Configuration Reference

To enable un-capped memory block allocation, include `write:global-max-blocks:-1` in the GCSFuse mount options:

### 1. Cloud Build Environment Variable:
```bash
_GCSFUSE_MOUNT_OPTIONS="implicit-dirs,client-protocol=grpc,write:enable-streaming-writes:true,write:global-max-blocks:-1"
```

### 2. Helm `--set` Flag:
```bash
--set gcsfuse.mountOptions="implicit-dirs\,client-protocol=grpc\,write:enable-streaming-writes:true\,write:global-max-blocks:-1"
```

### 3. GCSFuse Configuration File (`config.yaml`):
```yaml
write:
  enable-streaming-writes: true
  global-max-blocks: -1
```
