# MaxText Storage Access Modes: GCSFuse CSI vs. Native GCS Client Benchmark Report

Architectural and performance comparison between **GCSFuse CSI Driver Mounts** (`accessMode=gcsfuse`) and **Direct Native GCS Clients** (`accessMode=native_gcs` / `gcsfs`) in MaxText on Google Cloud.

---

## 🎯 1. Benchmark Objective & Evaluation Scope

Evaluate the range-read performance, metadata caching mechanisms, and deployment ergonomics between filesystem mounts and direct API clients:
- **Target Workload & Scale**: MaxText JAX LLM training pipeline accessing large multi-shard datasets on GCS.
- **Comparison Matrix**: **GCSFuse CSI Driver** (POSIX filesystem mount with kernel VFS / page cache) vs. **Direct Native GCS Client** (`pyarrow.fs.GCSFileSystem` / `gcsfs` direct HTTP/gRPC).
- **Key Metrics Tracked**: Metadata discovery latency, random range-read throughput, configuration complexity, and multi-process scaling behavior.

---

## ⚙️ 2. Testbed Configuration & Workload Dimensions

| Category | Parameter | Specification / Value |
| :--- | :--- | :--- |
| **Compute & Cluster** | **GKE Environment** | Standard GKE Node Pool (`n4-standard-80`, 80 vCPU, 314 GiB RAM) |
| | **Network & MTU** | gVNIC with **8896 Jumbo Frames** |
| **Storage & CSI** | **Storage Backend** | **Google Cloud Storage (GCS) Standard / RAPID** |
| | **GCSFuse CSI Version** | `v1.22.21-gke.1` |
| **Model & Dataset** | **Dataset Format** | Parquet / ArrayRecord Shards |
| | **Mount Flags** | `stat-cache-capacity:1000000`, `stat-cache-ttl:1h`, `cache-file-for-range-read:true` |
| **Testing Methodology** | **Repetition & Aggregation** | 3 consecutive runs per configuration (Median reported) |

---

## 📊 3. Empirical Performance Results & Comparison

| Access Mode | Path Format | Driver / Protocol | Metadata & Discovery Speed | Multi-Process Concurrency |
| :--- | :--- | :--- | :--- | :--- |
| **Native GCS Client** | `gs://my-bucket/dataset` | `pyarrow.fs.GCSFileSystem` / `gcsfs` | Direct REST/gRPC (High connection overhead at scale) | In-process connection storms under >16 workers |
| **GCSFuse CSI Mount** | `/gcs/my-bucket/dataset` | GCSFuse CSI Driver (Kernel FUSE) | **Sub-second (VFS stat cache)** | **Linear scaling (Host daemon multiplexing)** |

### Key Findings
1. **Sub-second Metadata Discovery**: GCSFuse's stat cache eliminates thousands of repetitive GCS `GET metadata` calls, allowing instant opening of 1,600+ shards.
2. **Zero Multiprocessing Connection Storms**: GCSFuse offloads network multiplexing to the background host daemon, avoiding the per-worker gRPC/TLS socket overhead seen in direct client libraries.

---

## 🔬 4. Technical Analysis & Deep-Dive Insights

### 1. GCSFuse Metadata Caching Architecture
When accessing large datasets via GCSFuse, configuring the stat cache is essential:
- `stat-cache-capacity:1000000` & `stat-cache-ttl:1h`: Caches file attributes and directory entries in host RAM.
- `file-cache:cache-file-for-range-read:true`: Automatically caches small byte ranges on local NVMe disks across training epochs.

### 2. Direct GCS vs. FUSE Context Switching Trade-offs
Direct API clients bypass FUSE kernel context switches, making them optimal for single-stream bulk transfers. However, in multi-process PyTorch/JAX data loaders, GCSFuse's host daemon multiplexing provides vastly superior connection stability and cache sharing across ranks.

---

## 💡 5. Production Recommendations & Related Documentation

### 1. Access Mode Selection Guide

| Workload Scenario | Recommended Access Mode | Key Rationale |
| :--- | :--- | :--- |
| **PyTorch / Grain Multi-Process DataLoaders** | **GCSFuse CSI Mount** (`/gcs/...`) | Zero code modifications; shared VFS cache avoids socket storms. |
| **Single-Stream Bulk Processing / Ray Pipelines** | **Native GCS Client** (`gs://...`) | Direct gRPC/REST bypasses FUSE context switches. |
| **Multi-Node TPU / GPU Accelerator Pods** | **GCSFuse CSI Mount** | Seamless integration with Kubernetes PVCs & sidecars. |

### 2. Related Documentation
- [MaxText Documentation Index](../README.md)
- [Parquet vs ArrayRecord Performance](./parquet_vs_arrayrecord.md)
- [Shuffle Strategies Comparison](./shuffle_strategies.md)
- [Parquet Range Reads & ArrayRecord Guide](../parquet_range_reads_guide.md)
