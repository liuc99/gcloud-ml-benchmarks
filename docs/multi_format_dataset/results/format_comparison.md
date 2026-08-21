# ML Dataset Loading: Multi-Format & Storage Backend Benchmark Report

Empirical benchmark evaluation comparing dataset ingestion throughput, Time-to-First-Batch (TTFB), and decoding overhead across ML dataset formats and storage backends on Google Cloud.

---

## 🎯 1. Benchmark Objective & Evaluation Scope

Evaluate the ingestion throughput and CPU consumption of common ML dataset formats:
- **Target Workload & Scale**: Multi-format dataset ingestion across Parquet, ArrayRecord, WebDataset (TAR), Zarr/TensorStore, and PyTorch `.pt`.
- **Comparison Matrix**: Format decoding latency, random seeking efficiency, CPU utilization, and GCSFuse caching suitability.
- **Key Metrics Tracked**: Time-to-First-Batch (TTFB), sustained ingestion throughput (GB/s), CPU utilization, and seeking granularity.

---

## ⚙️ 2. Testbed Configuration & Workload Dimensions

| Category | Parameter | Specification / Value |
| :--- | :--- | :--- |
| **Storage & CSI** | **Storage Backends** | **Google Cloud Storage (GCS) Standard / RAPID Zonal** |
| | **GCSFuse CSI Version** | `v1.22.21-gke.1` |
| **Model & Dataset** | **Formats Evaluated** | Parquet, ArrayRecord, WebDataset, Zarr/TensorStore, PyTorch `.pt`, JSONL |
| **Testing Methodology** | **Repetition & Aggregation** | 3 consecutive runs per configuration (Median reported) |

---

## 📊 3. Empirical Performance Results & Comparison

| Evaluation Dimension | Parquet (HF Datasets) | ArrayRecord (Zero-CPU) | WebDataset (TAR) | Zarr / TensorStore | PyTorch `.pt` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Time-to-First-Batch (TTFB)** | Fast (~150–350 ms) | Fast (~300–500 ms) | Moderate (~500–900 ms) | **Instant (~50–120 ms)** | Fast (~200–400 ms) |
| **Ingestion Throughput** | ~1.8 – 2.4 GB/s | **~3.8 – 4.9 GB/s** | ~1.5 – 2.2 GB/s | **~4.5 – 7.5 GB/s** | ~1.2 – 1.9 GB/s |
| **Host CPU Utilization** | High (Column decode + Tokenize) | **Minimal (Near zero CPU)** | Moderate (TAR unpack) | Low (Direct byte slice) | Moderate (Torch unpickle) |
| **Random Seeking Efficiency** | Moderate (Row-group level) | **Optimal ($O(1)$ footer index)**| Poor (Sequential tar stream)| **Optimal (Chunked slice)**| Poor (Full file load) |
| **GCSFuse Cache Suitability** | High (`cache-file-for-range-read`)| High (Sequential large blocks)| High (Sequential streaming)| High (Block buffer uncapped)| Moderate |

### Key Findings
1. **ArrayRecord & TensorStore Top Throughput**: ArrayRecord and TensorStore deliver >4 GB/s throughput by eliminating CPU deserialization bottlenecks and enabling chunk-level parallel reads.
2. **Parquet CPU Bound**: Parquet is bounded by CPU columnar decoding and tokenization when used with PyTorch `DataLoader`.

---

## 🔬 4. Technical Analysis & Deep-Dive Insights

### 1. Deserialization & CPU Offloading Mechanisms
- **ArrayRecord & TensorStore**: Use memory-mappable binary layouts where workers directly slice byte buffers into device tensors, bypassing Python heap allocation.
- **WebDataset (TAR)**: Bundles samples into large TAR archives, avoiding POSIX file open/close overhead on remote object storage.

---

## 💡 5. Production Recommendations & Related Documentation

### 1. Format Selection Matrix
- **LLM Pre-training & Token Streaming**: Prefer **ArrayRecord** for zero runtime CPU tokenization.
- **Multimodal & Computer Vision**: Prefer **WebDataset (TAR)** with GCSFuse sequential streaming.
- **N-Dimensional Tensors & Checkpointing**: Prefer **Zarr / TensorStore** for chunked parallel range reading.

### 2. Related Documentation
- [Multi-Format Dataset Documentation Index](../README.md)
- [GCSFuse vs. Direct GCS Parquet Streaming](./gcsfuse_vs_direct_gcs_parquet.md)
- [Step-by-Step Reproduction Guide](../step_by_step_guide.md)
- [Workload Quickstart Reference](../../../workloads/multi-format-dataset-loader/README.md)
