# PyTorch DDP + Storage Benchmark Suite

This directory contains the reproduction guides and performance results for PyTorch Distributed Data Parallel (DDP) model training and checkpointing benchmarks evaluating Google Cloud Managed Lustre, GCSFuse CSI, and native `gcsfs`.

---

## Directory Structure

```
docs/pytorch/
├── README.md                           # PyTorch workload overview & index
├── step_by_step_guide.md               # Reproduction guide for Managed Lustre, GCSFuse, and gcsfs
└── results/                            # Test results & storage backend comparisons
    ├── checkpoint_write_performance.md # Checkpoint pickling & network upload throughput (Decoupled write)
    ├── checkpoint_restore_performance.md # Checkpoint restore speed & unpickling comparison (Decoupled restore)
    └── rank_scaling_and_memory.md      # Rank topology & RAM OOM prevention
```

---

## Reproduction Guide

- [Step-by-Step Reproduction Guide](step_by_step_guide.md): Instructions for launching PyTorch DDP Llama 3.1 8B benchmark jobs on GKE against Managed Lustre (`LustreCsiDriver`), GCSFuse, and `gcsfs`.

---

## Benchmark Test Results

1. [Checkpoint Write Performance (Decoupled Save)](results/checkpoint_write_performance.md)
   - **Dimension**: Checkpoint serialization duration (CPU pickling) vs network storage streaming rates across GCSFuse CSI streaming writes and `gcsfs`.
   - **Highlights**: GCSFuse streaming write reached 611.51 MB/s (~2.6x faster than `gcsfs`).

2. [Checkpoint Restore Performance (Decoupled Load)](results/checkpoint_restore_performance.md)
   - **Dimension**: 45 GB (44.87 GiB) Pure Cold Checkpoint Restore speed across GCSFuse CSI Driver and Direct GCS (`gcsfs`).
   - **Highlights**: GCSFuse CSI restored in **28.25s ± 0.45s (1.59 GB/s effective)** under 3-run sampling; Direct GCS stabilized to **132.56s (0.34 GB/s)** via `fsspec.open` block caching.

3. [Rank Topology & Memory Scaling](results/rank_scaling_and_memory.md)
   - **Dimension**: Ranks per node scaling (2 ranks vs 4 ranks per node) and RAM OOM prevention.
   - **Highlights**: 2 ranks / node (~150 GB RAM peak) identified as the memory-safe sweet spot on 320 GB RAM `n4-standard-80` nodes.
