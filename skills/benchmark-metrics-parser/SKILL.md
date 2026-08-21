---
name: benchmark-metrics-parser
description: Sub-skill for parsing container stdout benchmark logs, extracting dataset load times, raw network throughput, and checkpoint durations, calculating multi-run statistical averages, and generating comparative Markdown reports.
---

# Benchmark Metrics Parser Sub-Skill (`benchmark-metrics-parser`)

This sub-skill parses container benchmark logs, extracts empirical throughput/latency metrics, calculates statistical metrics across multiple repeat runs (Mean, Min, Max, Standard Deviation), and formats comparative Markdown performance reports.

---

## 🛠️ Tasks & Capabilities

### 1. Extract Workload Container Output
Run `kubectl logs` to fetch stdout from the benchmark container:
```bash
kubectl logs job/${RELEASE_NAME}-workload-0 -c workload
```

### 2. Parse Metrics & Calculate Multi-Run Statistics
Pass raw single or multi-file log paths to `parse_metrics.py`:
```bash
# Single log parsing:
python3 skills/benchmark-metrics-parser/scripts/parse_metrics.py /path/to/log.txt

# Multi-file comparative parsing with automated Markdown table generation:
python3 skills/benchmark-metrics-parser/scripts/parse_metrics.py /path/to/lustre.log /path/to/gcsfuse.log /path/to/gcsfs.log --format=both --output-md=/path/to/report.md
```

The script extracts:
1. **HuggingFace DataLoader preparation time** (Manifest load & index build).
2. **Worker Spawn & Prefetch time** (`_PrefetchDataFetcher.__iter__` multi-process fork duration).
3. **Training step metrics** (Average / Min / Max step time, Average / Peak step throughput in samples/sec).
4. **Dual-Metric Checkpoint Saves**:
   - **Pure Storage I/O Throughput & Duration** (Physical network and disk upload rate, eliminating CPU serialization distortion).
   - **Total End-to-End Save Duration** (Full framework save hook latency including CPU pickling & barrier synchronization).
5. **Multi-Rank Sharded Checkpoint Aggregation** (Aggregates multi-process FSDP concurrent uploads into total physical cluster throughput).
6. **Checkpoint deletion latency**.
7. **MaxText / Multi-format dataset ingestion metrics** (TTFB, Ingestion Speed, Batch Latency p50/p95).

---

### 3. Format Comparative Matrix Reports

The script automatically extracts metrics and formats GitHub-style Markdown comparative matrix tables with dual-metric checkpoint separation:

| Metric / Dimension | Managed Lustre | GCSFuse CSI | Direct GCS (gcsfs) |
| :--- | :---: | :---: | :---: |
| **DataLoader Prep Time** | **2.2408s** | **0.8983s** | **4.3058s** |
| **Worker Spawn & Prefetch Time** | **20.88s** | **20.71s** | **21.52s** |
| **Avg Training Step Time** | **0.0425s** | **0.0452s** | **0.0487s** |
| **Avg Step Throughput** | **370.97 samples/s** | **354.19 samples/s** | **340.53 samples/s** |
| **Peak Step Throughput** | **528.92 samples/s** | **528.75 samples/s** | **517.27 samples/s** |
| **Checkpoint Size (per save)** | **44.82 GB** | **44.87 GB** | **44.87 GB** |
| **Pure Storage I/O Write Speed (DDP Single-Stream)** | **683.39 – 722.34 MB/s** | **442.67 – 500.25 MB/s** | **280.48 – 343.74 MB/s** |
| **Pure Storage I/O Write Speed (FSDP 4-Rank Concurrent)** | **2,850.20 MB/s (2.85 GB/s)** | **1,574.70 MB/s (1.57 GB/s)** | **1,438.50 MB/s (1.44 GB/s)** |
| **Mean Pure Storage Save Duration (45 GB)** | **16.04s (FSDP) / 67.23s (DDP)** | **28.52s (FSDP) / 91.86s (DDP)** | **31.20s (FSDP) / 138.68s (DDP)** |
| **Total Save Duration (incl. CPU serialization)** | **68.54s (DDP) / 88.24s (FSDP)** | **108.34s (DDP) / 100.83s (FSDP)** | **159.25s (DDP) / 103.52s (FSDP)** |
| **Checkpoint Delete Latency** | **4.245s** | **2.370s** | **0.260s** |

---

## 📜 MANDATORY BENCHMARK REPORT STANDARD SPECIFICATION (5-STAGE STRUCTURE)

All benchmark report documents (`docs/**/results/*.md`) generated or updated across `gcloud-ml-benchmarks` MUST strictly adhere to the following **5-Stage Architecture**:

```markdown
# [Workload / Scale] [Evaluation Dimension / Comparison] Benchmark Report

[1-2 sentence Executive Summary / TL;DR summarizing workload, comparison matrix, and core performance breakthrough]

---

## 🎯 1. Benchmark Objective & Evaluation Scope
- **Target Workload & Scale**: `<workload>` (e.g. 100GB Orbax Checkpoint, Llama 3.1 8B, 420GB Parquet).
- **Comparison Matrix**: `<baseline>` vs `<optimized>` (e.g. Un-rewritten Baseline vs. 1:1 Target-Aligned Resharding).
- **Key Metrics Tracked**: Primary latency/wall-time, effective storage throughput (MB/s), memory footprint (RAM), and numerical parity.

---

## ⚙️ 2. Testbed Configuration & Workload Dimensions
| Category | Parameter | Specification / Value |
| :--- | :--- | :--- |
| **Compute & Cluster** | **GKE Environment** | Standard GKE Node Pool (`<machine-type>`, `<vCPU>` vCPU, `<RAM>` RAM) |
| | **Network & MTU** | gVNIC with **8896 Jumbo Frames** (or MTU 1460) |
| **Storage & CSI** | **Storage Backend** | **Google Cloud Storage (GCS) RAPID Zonal / Standard** / **Managed Lustre** |
| | **GCSFuse CSI Version** | `<version>` (with flags e.g. `streaming-writes:true`) |
| **Model & Checkpoint / Dataset** | **Architecture / Format** | `<model>` / `<format>` |
| | **Scale & Dimensions** | Total volume, shard count, tensor shape, sequence length |
| | **Topology / Concurrency** | Number of worker processes, ranks per node, shuffle mode |
| **Testing Methodology** | **Repetition & Aggregation** | 3 consecutive runs per configuration (Median reported) |

---

## 📊 3. Empirical Performance Results & Acceleration
| Benchmark Evaluation Metric | Baseline / Variant A | Optimized / Variant B | Performance Gain & Impact |
| :--- | :--- | :--- | :--- |
| **[Primary Latency / Wall Time]** | **XX.XX s** | **YY.YY s** | **Z.ZZx Speedup (XX% time saved)** |
| **[Effective Storage Throughput]** | **XX.XX MB/s** | **YY.YY MB/s** | **+ZZ.ZZ MB/s gain** |
| **[Resource Footprint / RAM Peak]**| **XX.XX GB** | **YY.YY GB** | **Memory headroom / Zero OOM risk** |
| **[Data Parity / Loss-Free Check]** | N/A | **100% Loss-Free / Numerical Parity**| Lossless equivalent verification |

### Key Findings
1. **[Finding 1: Primary Speedup / Latency Breakdown]**: Concise mechanism explaining the wall-time difference.
2. **[Finding 2: Network / I/O Saturation]**: Throughput scaling and storage pipe saturation analysis.

---

## 🔬 4. Technical Analysis, Structural Breakdown & Deep-Dive Findings
- **Physical Layout & Chunk Geometry**: Structural breakdown table, directory hierarchy tree, and metadata inspection (`.zarray` / manifest).
- **Underlying Root Cause Mechanism**: I/O serialization, TCP socket contention, or memory buffer bounds.
- **Production ROI & Break-Even Amortization**: Cost/time tradeoff analysis (e.g. one-time CPU rewrite cost amortized over N restore cycles).

---

## 💡 5. Production Recommendations & Related Documentation
- **Production Selection Advice**: Practical decision criteria and deployment best practices.
- `[Workload Overview & Architecture] (<relative-path-to-workload-readme>)`
- `[Step-by-Step Reproduction Guide] (<relative-path-to-step-by-step-guide>)`
- `[Workload Helm Chart Reference] (<relative-path-to-helm-chart-readme>)`
```

---

## 🔒 MANDATORY ANONYMIZATION & QUALITY GUARDRAILS
1. **STRICT ZERO PRIVATE INFORMATION**:
   - NEVER include specific private cluster names (e.g. `*-gke-persistent`), internal GCP project IDs, personal GCS bucket names, or user emails.
   - ALWAYS use generic technical descriptions: `Standard GKE Node Pool (n4-standard-80, 80 vCPU, 314 GiB RAM)` and `gs://<user-bucket>/...`.
2. **STRICT ZERO REDUNDANCY**:
   - Do NOT repeat performance figures across multiple narrative paragraphs. State data clearly in tables, and keep prose focused on technical root causes.
3. **ALL-ENGLISH TABLES**:
   - All table headers, metric labels, and comparison summaries MUST be written in English.

