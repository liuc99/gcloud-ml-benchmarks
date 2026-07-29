# PyTorch DDP + Storage Benchmark Suite

This directory contains the reproduction guides and performance results for PyTorch Distributed Data Parallel (DDP) model training and checkpointing benchmarks evaluating **Google Cloud Managed Lustre**, **GCSFuse**, and **`gcsfs`**.

---

## 📁 Directory Structure

```
docs/pytorch/
├── README.md                           # PyTorch workload overview & index
├── step_by_step_guide.md               # Reproduction guide for Lustre, GCSFuse, and gcsfs
└── results/                            # Test results & storage backend comparisons
```

---

## 🛠️ Reproduction Guide

* [**Step-by-Step Reproduction Guide**](step_by_step_guide.md): Instructions for launching PyTorch DDP Llama 3.1 8B benchmark jobs on GKE against Managed Lustre (`LustreCsiDriver`), GCSFuse, and `gcsfs`.

---

## 📊 Benchmark Test Results

* Detailed comparative results across storage backends (Lustre vs GCSFuse vs gcsfs) will be populated here as benchmark runs complete.
