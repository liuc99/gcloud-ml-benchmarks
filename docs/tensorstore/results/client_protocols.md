# Benchmark Dimension: Client Protocols (HTTP/1.1 vs gRPC)

This document evaluates the trade-offs between **`client-protocol=http1`** (HTTP/1.1 REST API) and **`client-protocol=grpc`** (gRPC / HTTP/2 Direct Path) in GCSFuse when reading and writing multidimensional TensorStore Zarr arrays on GKE.

---

## 📊 Summary Performance Comparison Table

Measured across 32 nodes (128 worker ranks, 3.81 TB dataset per run):

| Client Protocol | Peak Read Throughput | Aggregate 3-Run Mean Read | Aggregate 3-Run Mean Write | Optimal Workload Path | Key Technical Drivers |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HTTP/1.1 (`http1`)** | **172,909.38 MB/s (1.35 Tbps)** | **142.80 ± 22.60 GB/s** | **94.91 ± 7.32 GB/s** | **READ HEAVY** (+22.3% faster Read than gRPC) | Un-multiplexed parallel TCP socket pool per sidecar; avoids Go gRPC channel lock contention. |
| **gRPC (`grpc`)** | **124,430.00 MB/s (995 Gbps)** | **116.73 ± 9.74 GB/s** | **107.84 ± 4.18 GB/s** | **WRITE HEAVY** (+13.6% faster Write than HTTP/1) | HTTP/2 streaming write buffer pipelining; higher write stability (±4.18 GB/s stddev). |

---

## 🔍 Architectural Analysis & Takeaways

### A. Why HTTP/1.1 Outperforms gRPC on Read Operations (+22.3%)

1. **Un-multiplexed Parallel TCP Sockets**:
   - GCSFuse HTTP/1.1 allocates an independent pool of dedicated TCP sockets per sidecar mounter daemon.
   - Each socket streams raw GCS HTTP GET payload buffers directly into application RAM without HTTP/2 stream framing overhead.

2. **Elimination of gRPC Stream Lock Contention**:
   - At 32-node scale (128 concurrent worker processes issuing parallel array range requests for 3.81 TB), gRPC multiplexes multiple logical requests over shared HTTP/2 channels.
   - Mutex lock contention inside the Go gRPC client stack creates read latency jitter under high concurrency, whereas HTTP/1.1 parallel sockets scale independently across multiple kernel TCP congestion windows.

### B. Why gRPC Outperforms HTTP/1.1 on Write Operations (+13.6%)

1. **HTTP/2 Streaming Write Pipelining**:
   - When writing sequential Zarr chunk files, gRPC pipelines chunk upload payload blocks over open HTTP/2 data streams without establishing new HTTP request headers for every chunk.
   - Combined with un-capped GCSFuse memory blocks (`write:global-max-blocks:-1`), gRPC achieves **107.84 GB/s aggregate write throughput** with high statistical consistency.

---

## 💡 Recommendation Matrix

- **Read-Dominant Workloads (Model Inference, Dataset Loading, Analytics)**: Use `--set gcsfuse.mountOptions="...,client-protocol=http1"`.
- **Write-Dominant Workloads (Model Checkpointing, Data Ingestion, Simulation Output)**: Use `--set gcsfuse.mountOptions="...,client-protocol=grpc"`.
