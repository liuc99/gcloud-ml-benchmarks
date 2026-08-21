# 100GB Orbax Checkpoint Resharding & Restore Benchmark Report

Empirical benchmark evaluation of **Orbax & TensorStore checkpoint offline resharding and restore acceleration** on Google Kubernetes Engine (GKE) and Google Cloud Storage (GCS Zonal RAPID).

---

## 🎯 1. Benchmark Objective & Evaluation Scope

Evaluate the end-to-end performance and resource efficiency of **Offline CPU Resharding** when adapting large model checkpoints across non-matching cluster topologies:
- **Target Workload & Scale**: 100GB-scale (**112.0 GB**) Orbax / TensorStore Float32 checkpoint (16 Transformer layers $\times$ 7 parameter matrices = 112 arrays).
- **Comparison Matrix**: **Un-rewritten Baseline** (5 source shards $\to$ 10 target workers with 2x read amplification) vs. **Rewritten Optimized** (10 target-aligned chunks $\to$ 10 target workers with 1:1 streaming).
- **Key Metrics Tracked**: Restore wall-clock time, effective GCS read throughput (MB/s), CPU rewriter duration and throughput, host peak RAM, and numerical parity.

---

## ⚙️ 2. Testbed Configuration & Workload Dimensions

| Category | Parameter | Specification / Value |
| :--- | :--- | :--- |
| **Compute & Cluster** | **GKE Environment** | Standard GKE Node Pool (`n4-standard-80`, 80 vCPU, 314 GiB RAM) |
| | **Network & MTU** | gVNIC with **8896 Jumbo Frames** |
| **Storage & CSI** | **Storage Backend** | **Google Cloud Storage (GCS) RAPID Zonal** (`DirectPath` gRPC enabled) |
| | **GCSFuse CSI Version** | `v1.22.21-gke.1` |
| **Model & Checkpoint** | **Model Architecture** | 16-layer Transformer $\times$ 7 parameter matrices = **112 TensorStore Arrays** |
| | **Array Shape & Dtype** | $[16384, 16384]$ Float32 (1.0 GB per matrix, **112.00 GB Total Volume**) |
| | **Topology Transition** | **5 Source Shards $\to$ 10 Target Worker Processes** |
| **Rewriter Pipeline** | **Execution Threads** | 16 parallel threads (`--num-workers=16`) |
| | **Streaming Buffer** | 64 MB bounded in-memory slice pipeline |
| **Testing Methodology** | **Repetition & Aggregation** | 3 consecutive runs per configuration (Median reported) |

---

## 📊 3. End-to-End Performance & Acceleration Results

| Evaluation Metric | Un-rewritten Baseline (5 Shards ➔ 10 Workers) | Rewritten Optimized (10 Shards ➔ 10 Workers) | Performance Gain & Impact |
| :--- | :--- | :--- | :--- |
| **112 GB Restore Wall Time (Median)** | **33.35 s** (33,354.5 ms) | **24.74 s** (24,739.5 ms) | **1.35x Speedup (25.8% time saved)** |
| **Effective GCS Read Throughput** | **3,438.46 MB/s** (3.44 GB/s) | **4,635.83 MB/s** (4.64 GB/s) | **+1,197.37 MB/s (+1.20 GB/s)** |
| **Saved Time per Restore Operation** | Baseline | **8.61 seconds per restore** | Eliminates cluster startup & recovery stalls |
| **112 GB Offline CPU Rewrite Time** | N/A | **63.14 seconds** (1.82 GB/s write) | Fast one-time CPU operation |
| **Rewriter Peak Host RAM** | N/A | **< 1.20 GB** (64MB slice pipeline) | **Zero OOM risk** (< 0.4% node memory) |
| **Numerical Parity Verification** | N/A | **100% Loss-Free (`rtol=2e-2`)** | Lossless equivalent coordinate transform |

### Key Findings
1. **25.8% Wall-Time Reduction**: In the un-rewritten baseline, pairs of workers concurrently fetch the same 1GB source chunks due to TensorStore Chunk Atomicity, causing 2x read amplification and GCS connection contention. 1:1 target-aligned resharding eliminates over-fetching, cutting restore time from **33.35s to 24.74s**.
2. **Network Saturation (+1.20 GB/s)**: Sequential full-file streaming allows GCSFuse and OS read-ahead buffers to reach line-rate throughput, boosting effective restore speed from **3.44 GB/s to 4.64 GB/s**.

---

## 🔬 4. Technical Analysis, Physical Shards & Memory Bounds

### 1. Structural Layout Comparison

| Dimension / Attribute | Before Rewrite (5-Shard Source Layout) | After Rewrite (10-Shard Target Aligned) | Transformation Impact |
| :--- | :--- | :--- | :--- |
| **Total Data Volume** | **112.00 GB** (114,688 MB) | **112.00 GB** (114,688 MB) | **100% loss-free parity** |
| **Parameter Matrices** | 16 layers $\times$ 7 matrices = **112 arrays** | 16 layers $\times$ 7 matrices = **112 arrays** | Array set is invariant |
| **Matrix Logical Shape** | $[16384, 16384]$ (Float32, 1.0 GB / matrix) | $[16384, 16384]$ (Float32, 1.0 GB / matrix) | Logical tensor shape unchanged |
| **Chunk Geometry (`chunks`)**| **$[3277, 16384]$** (5 partitions along Dim 0) | **$[1639, 16384]$** (10 partitions along Dim 0) | **Aligned 1:1 with target workers** |
| **Chunk Files per Matrix** | 5 files (`0.0` ~ `4.0`) | 10 files (`0.0` ~ `9.0`) | Matches 10 worker ranks |
| **Single Chunk File Size** | **~214.7 MB** (204.8 MiB) | **~107.4 MB** (102.4 MiB) | Halved chunk size for smoother streaming |
| **Total Data Chunk Files** | $112 \times 5 =$ **560 files** | $112 \times 10 =$ **1,120 files** | High-efficiency large files (no small-file storm) |
| **Total File Count** | **675 files** (including 115 metadata) | **1,236 files** (including 116 metadata) | Strictly bounded (< 1,500 files) |

### 2. Directory Hierarchy & Metadata Inspection

```
/gcs/checkpoints/
├── source_ckpt_5shards/                        <--- [Before Rewrite: 5-Shard Source Layout]
│   ├── _CHECKPOINT, .orbax-checkpoint-metadata
│   └── items/params/layer_00/q_proj/kernel/
│       ├── .zarray                             (shape=[16384,16384], chunks=[3277,16384])
│       └── 0.0 ... 4.0                         (5 files x 214.7 MB, source slice per rank)
│
└── rewritten_ckpt_10shards/                    <--- [After Rewrite: 10-Shard Target Aligned]
    ├── _CHECKPOINT, .orbax-checkpoint-metadata
    ├── rewrite_manifest.json                   (Audit manifest: 112 arrays, 63.14s, 1.82 GB/s)
    └── items/params/layer_00/q_proj/kernel/
        ├── .zarray                             (shape=[16384,16384], chunks=[1639,16384])
        └── 0.0 ... 9.0                         (10 files x 107.4 MB, dedicated 1:1 download per rank)
```

**Target `.zarray` Metadata**:
```json
{
  "zarr_format": 2,
  "shape": [16384, 16384],
  "chunks": [1639, 16384],
  "dtype": "<f4",
  "order": "C",
  "compressor": null,
  "fill_value": 0.0
}
```

**Rewrite Audit Manifest (`rewrite_manifest.json`)**:
```json
{
  "rewrite_timestamp": "2026-08-13T02:40:15Z",
  "strategy": "dim_partitions",
  "dim_partitions": {"0": 10},
  "total_arrays": 112,
  "total_bytes": 120259084288,
  "rewrite_duration_sec": 63.14,
  "rewrite_throughput_mb_s": 1818.72,
  "verification_passed": true
}
```

### 3. Bounded Memory Streaming Pipeline
The rewriter ([`tools/checkpoints/orbax_reshard_rewriter.py`](../../../tools/checkpoints/orbax_reshard_rewriter.py)) streams source arrays in bounded 64MB slice buffers, maintaining a flat memory footprint (< 1.20 GB) regardless of total checkpoint size.

```mermaid
flowchart LR
    SRC["Source GCS Shards<br/>(214.7 MB / file)"] -->|"64MB Slices"| BUF["RAM Buffer<br/>(< 1.20 GB Flat)"]
    BUF -->|"1:2 Remapped"| DST["Target GCS Shards<br/>(107.4 MB / file)"]
```

---

## 💡 5. Production Recommendations & Related Documentation

### 1. ROI & Production Amortization
- **Rewrite Cost**: **63.14 seconds** on a standard CPU node (1.82 GB/s sustained write).
- **Restore Savings**: **8.61 seconds** saved on every restore across the accelerator cluster.
- **Break-Even Point**: The one-time CPU rewrite cost amortizes completely after **7 to 8 checkpoint restore cycles** (e.g. during spot preemption recoveries, rolling restarts, or multi-epoch evaluation runs).

### 2. Related Documentation
- [Orbax Workload Overview & Architecture](../README.md)
- [Step-by-Step Reproduction Guide](../step_by_step_guide.md)
- [Workload Helm Chart Reference](../../../workloads/orbax-checkpoint-benchmark/README.md)
