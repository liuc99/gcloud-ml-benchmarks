# PyTorch DDP Llama 3.1 8B Storage & Checkpoint Benchmark (`hf-pytorch-lightning-cpu`)

A distributed machine learning workload simulating **PyTorch Distributed Data Parallel (DDP) Llama 3.1 8B** training steps and large-scale model state dict checkpoint saving (45 GB) across **Google Cloud Managed Lustre**, **Google Cloud Storage (GCSFuse)**, and native **`gcsfs`**.

---

## 🚀 Key Performance Highlights (45 GB Model Checkpointing)

| Storage Backend | Checkpoint Serialization & Write Time | Sustained Network Write Throughput | Stall Time Reduction |
| :--- | :--- | :--- | :--- |
| **GCSFuse Streaming Writes** | **78.42 seconds** | **611.51 MB/s** | **-67.1% Save Stall** (~3.0x faster save) |
| **Direct GCS Client (`gcsfs`)** | **238.53 seconds** | **192.83 MB/s** | Baseline |

> 📊 For full empirical data and analysis, see [Checkpoint Write Performance](../../docs/pytorch/results/checkpoint_write_performance.md), [Checkpoint Restore Performance](../../docs/pytorch/results/checkpoint_restore_performance.md), and [Rank Topology Scaling](../../docs/pytorch/results/rank_scaling_and_memory.md).

---

## ☸️ Helm Chart Parameters Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `workload.nodes` | int | `2` | Number of distributed compute nodes in DDP group. |
| `workload.ranksPerNode` | int | `2` | Worker ranks per node (`torchrun --nproc_per_node`). |
| `workload.backend` | string | `"lustre"` | Target storage backend under test (`lustre`, `gcsfuse`, `gcsfs`). |
| `workload.modelName` | string | `"meta-llama/Meta-Llama-3.1-8B"` | Model architecture simulated for tensor state dicts. |
| `workload.maxSteps` | int | `100` | Total training iterations before checkpoint trigger. |
| `lustre.enabled` | bool | `true` | Enables Managed Lustre CSI Driver PVC mount (`/lustre`). |
| `lustre.checkpointPvc` | string | `"lustre-pvc"` | PVC name for Managed Lustre instance. |
| `gcsfuse.enabled` | bool | `false` | Enables GCSFuse CSI Driver sidecar mount (`/gcs`). |
| `gcsfuse.checkpointBucket` | string | `""` | Target GCS bucket for checkpoint writes. |

---

## 🚀 Quickstart Deployment

### Scenario A: Google Cloud Managed Lustre
```bash
helm install pytorch-ddp-lustre workloads/hf-pytorch-lightning-cpu/helm_chart -f workloads/hf-pytorch-lightning-cpu/helm_chart/values_base.yaml \
  --set workload.nodes=2 \
  --set workload.ranksPerNode=2 \
  --set workload.backend="lustre" \
  --set lustre.enabled=true \
  --set lustre.checkpointPvc="lustre-pvc"
```

### Scenario B: GCSFuse Streaming Writes
```bash
helm install pytorch-ddp-gcsfuse workloads/hf-pytorch-lightning-cpu/helm_chart -f workloads/hf-pytorch-lightning-cpu/helm_chart/values_base.yaml \
  --set workload.nodes=2 \
  --set workload.ranksPerNode=2 \
  --set workload.backend="gcsfuse" \
  --set gcsfuse.enabled=true \
  --set gcsfuse.checkpointBucket="<YOUR_BUCKET>" \
  --set-string gcsfuse.mountOptions="implicit-dirs\,write:enable-streaming-writes:true\,write:global-max-blocks:-1"
```

### Scenario C: Native Python `gcsfs` Client
```bash
helm install pytorch-ddp-gcsfs workloads/hf-pytorch-lightning-cpu/helm_chart -f workloads/hf-pytorch-lightning-cpu/helm_chart/values_base.yaml \
  --set workload.nodes=2 \
  --set workload.ranksPerNode=2 \
  --set workload.backend="gcsfs" \
  --set gcsfs.ckptWritePath="gs://<YOUR_BUCKET>/checkpoints_gcsfs"
```

---

## 📚 Complete Documentation Suite

- [PyTorch Documentation Index](../../docs/pytorch/README.md)
- [Step-by-Step Reproduction Guide](../../docs/pytorch/step_by_step_guide.md)
- [Model Checkpoint Write Performance](../../docs/pytorch/results/checkpoint_write_performance.md)
- [Model Checkpoint Restore Performance](../../docs/pytorch/results/checkpoint_restore_performance.md)
- [Rank Topology & Memory Scaling](../../docs/pytorch/results/rank_scaling_and_memory.md)
