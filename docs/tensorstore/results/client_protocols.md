# TensorStore Client Protocols: HTTP/1.1 vs. gRPC Benchmark Report

Empirical benchmark evaluation comparing **`client-protocol=http1`** (HTTP/1.1 REST API) and **`client-protocol=grpc`** (gRPC / HTTP/2 DirectPath) in GCSFuse when streaming multidimensional TensorStore Zarr arrays on GKE.

---

## 🎯 1. Benchmark Objective & Evaluation Scope

Evaluate protocol performance trade-offs between multiplexed gRPC and parallel HTTP/1.1 sockets:
- **Target Workload & Scale**: TensorStore multidimensional arrays (3.81 TB total dataset) across 32 GKE nodes (128 worker ranks).
- **Comparison Matrix**: **HTTP/1.1 (`http1`)** vs. **gRPC (`grpc`)** with GCSFuse CSI Driver.
- **Key Metrics Tracked**: Peak read throughput, 3-run mean read/write throughput (GB/s), and statistical standard deviation.

---

## ⚙️ 2. Testbed Configuration & Workload Dimensions

| Category | Parameter | Specification / Value |
| :--- | :--- | :--- |
| **Compute & Cluster** | **GKE Environment** | Standard GKE Node Pools (32 nodes, `n4-standard-80`, 50 Gbps gVNIC) |
| | **Network & MTU** | gVNIC with **8896 Jumbo Frames** |
| **Storage & CSI** | **Storage Backend** | **Google Cloud Storage (GCS) RAPID Zonal** |
| | **GCSFuse CSI Version** | `v1.22.21-gke.1` |
| **Model & Checkpoint** | **Dataset Scale** | 3,814.72 GB (3.81 TB) per run across 128 ranks |
| **Testing Methodology** | **Repetition & Aggregation** | 3 consecutive runs per configuration (Mean ± StdDev reported) |

---

## 📊 3. Empirical Performance Results & Comparison

| Client Protocol | Peak Read Throughput | Aggregate Mean Read (3 Runs) | Aggregate Mean Write (3 Runs) | Optimal Workload Path | Performance Summary |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HTTP/1.1 (`http1`)** | **172,909.38 MB/s (1.35 Tbps)** | **142.80 ± 22.60 GB/s** | **94.91 ± 7.32 GB/s** | **Read Heavy** | **+22.3% faster Read than gRPC**; Un-multiplexed parallel TCP sockets. |
| **gRPC (`grpc`)** | **124,430.00 MB/s (995 Gbps)** | **116.73 ± 9.74 GB/s** | **107.84 ± 4.18 GB/s** | **Write Heavy** | **+13.6% faster Write than HTTP/1**; HTTP/2 streaming write buffers. |

### Key Findings
1. **HTTP/1.1 Wins on Reads (+22.3%)**: Reached peak **1.35 Tbps (172.9 GB/s)** by leveraging independent parallel TCP sockets and avoiding Go gRPC channel mutex locks.
2. **gRPC Wins on Writes (+13.6%)**: Reached **107.84 GB/s write rate** with high statistical stability (±4.18 GB/s stddev) via HTTP/2 stream pipelining.

---

## 🔬 4. Technical Analysis & Deep-Dive Insights

### 1. HTTP/1.1 Read Advantage
GCSFuse HTTP/1.1 creates independent dedicated TCP socket pools per sidecar. Each socket streams raw GCS HTTP GET payload buffers without HTTP/2 stream framing overhead or mutex lock contention.

### 2. gRPC Write Streaming Efficiency
gRPC pipelines chunk upload blocks over open HTTP/2 data streams without establishing new HTTP request headers for each chunk, sustaining high write bandwidth.

---

## 💡 5. Production Recommendations & Related Documentation

### 1. Protocol Recommendation Matrix
- **Read-Dominant Workloads (Model Inference, Dataset Loading)**: Use `client-protocol=http1`.
- **Write-Dominant Workloads (Model Checkpointing, Data Ingestion)**: Use `client-protocol=grpc`.

### 2. Related Documentation
- [Multi-Node Cluster Scaling](./node_scaling.md)
- [Network MTU Impact](./network_mtu.md)
- [TensorStore Overview](../README.md)
