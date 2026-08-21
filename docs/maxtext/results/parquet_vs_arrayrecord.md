# MaxText Ingestion: Parquet vs. ArrayRecord Benchmark Report

Empirical benchmark evaluation comparing un-tokenized columnar **Parquet** (with runtime CPU tokenization and GCS range reads) against pre-tokenized binary **ArrayRecord** for MaxText LLM training pipelines on Google Cloud.

---

## 🎯 1. Benchmark Objective & Evaluation Scope

Evaluate the ingestion throughput, storage footprint, and CPU overhead of dataset formats for large-scale LLM training:
- **Target Workload & Scale**: MaxText JAX LLM training pipeline ingesting 1,650 dataset shards (420.10 GB raw Parquet vs. 155.27 GB ArrayRecord).
- **Comparison Matrix**: **Parquet** (String columnar format requiring runtime BPE/WordPiece tokenization) vs. **ArrayRecord** (Pre-tokenized `int32` dense token sequences).
- **Key Metrics Tracked**: Total storage volume, cold start Time-to-First-Batch (TTFB), per-step batch latency (p50/p99), CPU tokenization overhead, and effective tensor throughput.

---

## ⚙️ 2. Testbed Configuration & Workload Dimensions

| Category | Parameter | Specification / Value |
| :--- | :--- | :--- |
| **Compute & Cluster** | **GKE Environment** | Standard GKE Node Pool (`n4-standard-80`, 80 vCPU, 314 GiB RAM) |
| | **Network & MTU** | gVNIC with **8896 Jumbo Frames** |
| **Storage & CSI** | **Storage Backend** | **Google Cloud Storage (GCS) Standard / RAPID** |
| | **GCSFuse CSI Version** | `v1.22.21-gke.1` |
| **Model & Dataset** | **Dataset Dimensions** | 1,650 Shards (420.10 GB Parquet vs. 155.27 GB ArrayRecord) |
| | **Batch Size & Sequence**| `batch_size=128`, `sequence_length=2048` (`int32` token arrays) |
| | **Shuffle Strategy** | Two-Stage Shuffle (8 stream interleaving, 20,000-sample sliding buffer) |
| **Testing Methodology** | **Repetition & Aggregation** | 3 consecutive runs per configuration (Median reported) |

---

## 📊 3. Empirical Performance Results & Acceleration

| Benchmark Evaluation Metric | Parquet (Un-tokenized) | ArrayRecord (Pre-tokenized) | Performance Gain & Impact |
| :--- | :--- | :--- | :--- |
| **Storage & Transfer Size** | **420.10 GB** (100%) | **155.27 GB** (**36.1%**) | **63.9% reduction in storage cost & egress** |
| **Training CPU Overhead** | High (79% ~ 92% of step duration) | **Zero-CPU Ingestion** | Eliminates host CPU tokenization bottleneck |
| **Cold Start / TTFB** | **7.28 s** (20,000 sample buffer priming) | **3.53 s** (Direct binary read) | **51.5% reduction in startup stall** |
| **Steady-State Step Latency (Avg)** | **23.76 ms** (With 8-thread tokenizer) | **19.15 ms** (Direct tensor mapping) | **19.4% reduction in per-step batch latency** |
| **Tail Latency (p99)** | High jitter (Up to 290 ms) | **Stable (20–38 ms)** | Eliminates distributed stragglers |
| **Effective Tensor Throughput** | **3,287 samples/s** (25.68 MB/s) | **4,888 samples/s** (38.19 MB/s) | **+48.7% higher effective ingestion rate** |

### Key Findings
1. **63.9% Storage & Bandwidth Reduction**: Pre-tokenizing text into dense 32-bit integer arrays (`int32`) shrinks dataset volume from **420.10 GB to 155.27 GB**, cutting GCS network egress and storage charges by nearly two-thirds.
2. **Zero-CPU Ingestion**: ArrayRecord reads pre-tokenized binary buffers directly into JAX device memory, eliminating the 79%~92% host CPU tokenization bottleneck and boosting tensor throughput by **+48.7%**.

---

## 🔬 4. Technical Analysis & Deep-Dive Insights

### 1. The Host CPU Tokenization Bottleneck
When streaming raw Parquet during training, the host CPU must decode columnar pages, reconstruct Python strings, and execute tokenizer algorithms (BPE / WordPiece) across 80 threads. This creates severe CPU core contention and high tail latency (p99 up to 290 ms).

### 2. Binary Sequential Direct Mapping
ArrayRecord stores fixed-length integer arrays sequentially with optimized chunk indexing. Worker processes execute direct binary memory-mapped reads without CPU string parsing, achieving flat, predictable step times (~19 ms).

---

## 💡 5. Production Recommendations & Related Documentation

### 1. Selection & Best Practices
- **For Large-Scale Training (TPU/GPU Pods)**: Always pre-tokenize datasets into **ArrayRecord** format offline before training.
- **For Rapid Prototyping & Ad-hoc Exploration**: Parquet remains suitable for small-scale experiments where offline conversion overhead is undesirable.

### 2. Related Documentation
- [MaxText Documentation Index](../README.md)
- [Shuffle Strategies Comparison](./shuffle_strategies.md)
- [Storage Access Modes Evaluation](./storage_access_modes.md)
- [Parquet Range Reads & ArrayRecord Guide](../parquet_range_reads_guide.md)
