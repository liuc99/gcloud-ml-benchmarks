# GCSFuse Memory Block Buffer Tuning (`write:global-max-blocks`) Benchmark Report

Empirical benchmark evaluation measuring the throughput and memory dynamics of **Un-Capped Memory Blocks (`write:global-max-blocks:-1`) vs. Default Memory Caps** during TensorStore operations over GCSFuse on GKE.

---

## 🎯 1. Benchmark Objective & Evaluation Scope

Evaluate GCSFuse write streaming buffer dynamics and backpressure alleviation:
- **Target Workload & Scale**: Single-node TensorStore Zarr array (128 GB volume, 8 worker processes, 200 MB chunks) on `n4-standard-80`.
- **Comparison Matrix**: **Un-Capped (`write:global-max-blocks:-1`)** vs. **Default Capped Memory Blocks**.
- **Key Metrics Tracked**: Aggregate write throughput (MB/s), aggregate read throughput (MB/s), and write speedup.

---

## ⚙️ 2. Testbed Configuration & Workload Dimensions

| Category | Parameter | Specification / Value |
| :--- | :--- | :--- |
| **Compute & Cluster** | **GKE Environment** | Standard GKE Node Pool (`n4-standard-80`, 80 vCPU, 314 GiB RAM) |
| | **Network & MTU** | gVNIC with **8896 Jumbo Frames** |
| **Storage & CSI** | **Storage Backend** | **Google Cloud Storage (GCS) RAPID Zonal** |
| | **GCSFuse CSI Version** | `v1.22.21-gke.1` |
| **Model & Checkpoint** | **Array Volume** | 128 GB Float32 TensorStore Array (8 workers) |
| **Testing Methodology** | **Repetition & Aggregation** | 3 consecutive runs per configuration (Median reported) |

---

## 📊 3. Empirical Performance Results & Comparison

| Mount Option State | Parameter Configuration | Aggregate Write Throughput | Aggregate Read Throughput | Write Speedup & Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Un-Capped (`ON`)** | `write:global-max-blocks:-1` | **3,820.93 MB/s (~30.57 Gbps)** | **4,558.31 MB/s (~36.47 Gbps)** | **+107% Write Speedup**; Eliminates thread backpressure. |
| **Default Capped (`OFF`)** | Default Memory Blocks | **1,850.00 MB/s (~14.80 Gbps)** | **2,749.70 MB/s (~22.00 Gbps)** | **Throttled Baseline (-51.6% penalty)**; Buffer exhaustion stalls threads. |

### Key Findings
1. **+107% Write Throughput Surge**: Setting `write:global-max-blocks:-1` doubles write throughput from **1,850.00 MB/s to 3,820.93 MB/s** on a single node.
2. **Eliminates Write Backpressure Loops**: Prevents GCSFuse streaming queues from exhausting block tokens during concurrent chunk flushing.

---

## 🔬 4. Technical Analysis & Deep-Dive Insights

### 1. The Block Token Exhaustion Bottleneck
By default, GCSFuse limits streaming buffer blocks. Concurrent multi-worker chunk writes rapidly exhaust tokens, forcing worker threads into sleep loops.

### 2. Dynamic Memory Buffer Pipelining
Un-capping block allocation allows the GCSFuse daemon to dynamically allocate RAM buffers, keeping network upload pipelines saturated.

---

## 💡 5. Production Recommendations & Related Documentation

### 1. Configuration Best Practices

```yaml
# Recommended GCSFuse config.yaml
write:
  enable-streaming-writes: true
  global-max-blocks: -1
```

```bash
# Recommended Mount Options
_GCSFUSE_MOUNT_OPTIONS="implicit-dirs,client-protocol=grpc,write:enable-streaming-writes:true,write:global-max-blocks:-1"
```

### 2. Related Documentation
- [Multi-Node Cluster Scaling](./node_scaling.md)
- [Client Protocols Evaluation](./client_protocols.md)
- [TensorStore Overview](../README.md)
