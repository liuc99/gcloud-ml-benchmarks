# PyTorch DDP + Storage Benchmark Suite

This directory contains the reproduction guides and performance results for PyTorch Distributed Data Parallel (DDP) model training and checkpointing benchmarks evaluating **Google Cloud Managed Lustre**, **GCSFuse**, and **`gcsfs`**.

---

## 📁 Directory Structure

```
docs/pytorch/
├── README.md                           # PyTorch workload overview & index
├── step_by_step_guide.md               # Reproduction guide for Lustre, GCSFuse, and gcsfs
└── results/                            # Test results & storage backend comparisons
    ├── storage_backends.md             # Managed Lustre vs GCSFuse vs Direct GCS (gcsfs)
    ├── checkpoint_performance.md       # Checkpoint pickling & network upload throughput
    └── rank_scaling_and_memory.md      # Rank topology & RAM OOM prevention
```

---

## 🛠️ Reproduction Guide

* [**Step-by-Step Reproduction Guide**](step_by_step_guide.md): Instructions for launching PyTorch DDP Llama 3.1 8B benchmark jobs on GKE against Managed Lustre (`LustreCsiDriver`), GCSFuse, and `gcsfs`.

---

## 📊 Benchmark Test Results

1. [**Storage Backends Comparison**](results/storage_backends.md)
   * **Dimension**: Managed Lustre vs GCSFuse Streaming Writes vs Direct GCS (`gcsfs`).
   * **Highlights**: Managed Lustre achieved **953.41 MB/s** checkpoint write speed (~4.5x faster save than GCS), GCSFuse streaming write reached **611.51 MB/s**.

2. [**Model Checkpointing Performance**](results/checkpoint_performance.md)
   * **Dimension**: Checkpoint serialization duration (CPU pickling) vs network streaming rates.
   * **Highlights**: Complete 45 GB (44.87 GiB) checkpoint save in **53.16s** on Lustre vs 238.53s on `gcsfs` (77.7% stall time reduction).

3. [**Rank Topology & Memory Scaling**](results/rank_scaling_and_memory.md)
   * **Dimension**: Ranks per node scaling (2 ranks vs 4 ranks per node) and RAM OOM prevention.
   * **Highlights**: 2 ranks / node (~150 GB RAM peak) identified as the memory-safe sweet spot on 320 GB RAM `n4-standard-80` nodes.

