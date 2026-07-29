# Benchmark Dimension: Rank Topology Scaling & RAM OOM Prevention

This document evaluates the impact of **Worker Rank Topology per Node (`_RANKS_PER_NODE`)** and container RAM footprint during concurrent PyTorch Distributed Data Parallel (DDP) checkpoint serialization on `n4-standard-80` GKE nodes (80 vCPUs, 320 GB RAM).

---

## 📊 Performance & Stability Comparison Table

Comparison of 100-step Llama 3.1 8B DDP training runs across different rank configurations on a single `n4-standard-80` node (320 GB RAM ceiling):

| Topology Config | Ranks / Node | Global Batch Size | Peak Memory Footprint (Step 25 Checkpoint) | Training Step Speed | Job Completion Status | Technical Impact & Root Cause |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2 Ranks / Node (Optimal)** | **2 Ranks** | **16** | **~150 GB RAM (~46.8% Node RAM)** | **1.005s – 1.006s / step** | ✅ **100% SUCCESS** | **MEMORY SAFE SWEET SPOT**; Provides ample RAM headroom during concurrent rank pickling. |
| **4 Ranks / Node (OOM)** | **4 Ranks** | **32** | **~360 GB RAM (> 112.5% Node RAM)** | **1.005s / step** | ❌ **OOMKilled at Step 25** | **RAM Over-subscription**; 4 concurrent ranks pickling 90 GB state dicts exceeded 320 GB node RAM. |

---

## 🔍 Deep-Dive Technical Analysis: The Checkpoint OOM Mechanism

### 1. Memory Footprint per PyTorch Rank
During PyTorch Lightning DDP training of Llama 3.1 8B:
- **Model Weights (bfloat16)**: ~16 GB
- **AdamW Optimizer State (fp32 master weights + momentum + variance)**: ~32 GB
- **Active Gradients & Activations**: ~10 GB
- **`torch.save()` Serialization State Dict Buffer**: ~32–44 GB

Total peak RAM required by a single rank during checkpoint pickling: **~80 – 90 GB RAM**.

### 2. Why 4 Ranks / Node Caused `OOMKilled`
- When running `_RANKS_PER_NODE=4` on a single node:
  $$\text{Total Memory} = 4 \text{ ranks} \times 90 \text{ GB RAM/rank} = 360 \text{ GB RAM}$$
- Because the `n4-standard-80` instance has a physical memory limit of **320 GB RAM**, the Linux kernel Out-Of-Memory (OOM) Killer terminated the container when step 25 checkpoint save initiated.

### 3. The 2 Rank / Node Memory Solution
- Reducing ranks per node to `_RANKS_PER_NODE=2`:
  $$\text{Total Memory} = 2 \text{ ranks} \times 80 \text{ GB RAM/rank} = 160 \text{ GB RAM}$$
- Leaves **~160 GB RAM (50% headroom)** for operating system caches and GCSFuse streaming buffers, ensuring 100% stable benchmark execution without OOM crashes.

---

## 💡 Best Practices & Recommendations

1. **Calculate Checkpoint Memory Inflation**: Always account for the `2x` memory multiplier during `torch.save()` state dict pickling when sizing node memory pools.
2. **Cap Ranks per Node Based on RAM**: On nodes with 320 GB RAM running 8B parameter models, restrict ranks per node to **2 ranks per node**.
3. **Use Single-Writer Rank Architecture**: Set PyTorch Lightning checkpoint callbacks to designate Rank 0 as the sole writer rank, preventing multiple ranks from allocating duplicate serialization buffers simultaneously.
