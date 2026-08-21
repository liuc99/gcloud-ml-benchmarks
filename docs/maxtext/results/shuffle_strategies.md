# MaxText Shuffle Strategies: None vs. Two-Stage vs. Global Shuffle Benchmark Report

Empirical benchmark evaluation analyzing cold-start startup delay, Time-to-First-Batch (TTFB), and steady-state batch latency across dataset shuffle strategies in MaxText on Google Cloud.

---

## 🎯 1. Benchmark Objective & Evaluation Scope

Evaluate the trade-offs between startup latency, index scanning overhead, and randomness across shuffle modes:
- **Target Workload & Scale**: MaxText JAX LLM training pipeline ingesting 1,600+ dataset shards in Parquet and ArrayRecord formats.
- **Comparison Matrix**: **`none`** (Sequential streaming) vs. **`two_stage`** (Sliding in-memory window buffer) vs. **`global`** (Global random index sampling across all shards).
- **Key Metrics Tracked**: Time-to-First-Batch (TTFB), upfront index scanning penalty, and steady-state batch latency (p50/p95).

---

## ⚙️ 2. Testbed Configuration & Workload Dimensions

| Category | Parameter | Specification / Value |
| :--- | :--- | :--- |
| **Compute & Cluster** | **GKE Environment** | Standard GKE Node Pool (`n4-standard-80`, 80 vCPU, 314 GiB RAM) |
| | **Network & MTU** | gVNIC with **8896 Jumbo Frames** |
| **Storage & CSI** | **Storage Backend** | **Google Cloud Storage (GCS) Standard / RAPID** |
| | **GCSFuse CSI Version** | `v1.22.21-gke.1` |
| **Model & Dataset** | **Dataset Scale** | 1,600+ Shards (~420 GB Parquet / ~155 GB ArrayRecord) |
| | **Batch Configuration** | `batch_size=128`, `sequence_length=2048` |
| | **Sliding Buffer** | 20,000 samples for `two_stage` shuffle |
| **Testing Methodology** | **Repetition & Aggregation** | 3 consecutive runs per configuration (Median reported) |

---

## 📊 3. Empirical Performance Results & Acceleration

| Format & Shuffle Strategy | Time to First Batch (TTFB) | Upfront Index Scanning Penalty | Batch Latency (p50 / p95) | Main Characteristic & Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Parquet (`none` / `two_stage`)** | **~372 ms** | **0 ms** | ~1.5 ms / 4.2 ms | Instant start; zero pre-processing waiting time. |
| **Parquet (`global` shuffle)** | **91.64 s** | **91.47 s** | ~0.01 ms | **Severe 91s cold-start penalty** scanning 1600+ Parquet footers. |
| **ArrayRecord (`none` / `two_stage`)**| ~4.7 s – 7.0 s | **0 ms** | **0.34 ms / 0.57 ms** | Sub-millisecond batch latency, zero runtime CPU tokenization. |
| **ArrayRecord (`global` shuffle)** | **~6.5 s** | **31.56 ms** | **0.33 ms / 0.48 ms** | **2900x faster index loading** than Parquet global shuffle. |

### Key Findings
1. **91s Parquet Global Shuffle Freeze**: Global shuffle requires scanning the footer metadata of all 1,600+ Parquet files before emitting the first batch, creating an upfront 91.47s idle delay.
2. **2900x Faster Index Loading with ArrayRecord**: ArrayRecord's binary footer indexing loads in only **31.56 ms**, making global shuffle practical without blocking training startup.

---

## 🔬 4. Technical Analysis & Deep-Dive Insights

### 1. Parquet Footer Metadata Storm
In Parquet, sample row-counts and dictionary offsets reside in Thrift footer blocks. Reading footers across 1,600+ remote GCS files triggers 1,600+ serialized HTTP range requests, causing high network round-trip overhead.

### 2. Two-Stage Sliding Window Buffer
`two_stage` shuffle interleaves active streams across shards and samples from a sliding in-memory queue (e.g. 20,000 samples). This delivers continuous streaming throughput without any upfront index scanning stall.

---

## 💡 5. Production Recommendations & Related Documentation

### 1. Production Selection Guide
- **Production Standard**: Use **`two_stage`** shuffle for large-scale distributed training to guarantee zero-overhead instantaneous startup with sufficient sample randomness.
- **Avoid Global Shuffle on Parquet**: Never enable `global` shuffle directly on large multi-shard Parquet datasets stored on cloud object storage.

### 2. Related Documentation
- [MaxText Documentation Index](../README.md)
- [Parquet vs ArrayRecord Performance](./parquet_vs_arrayrecord.md)
- [Storage Access Modes Evaluation](./storage_access_modes.md)
- [Parquet Range Reads & ArrayRecord Guide](../parquet_range_reads_guide.md)
