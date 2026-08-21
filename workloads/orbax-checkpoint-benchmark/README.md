# Orbax Checkpoint Resharding & Restore Benchmark (`orbax-checkpoint-benchmark`)

A cloud-native workload and benchmark harness for **Orbax & TensorStore checkpoint offline resharding, topology adaptation, and restore acceleration** on Google Cloud Platform (GCP), Google Kubernetes Engine (GKE), and Google Cloud Storage (GCS / GCSFuse).

---

## 🚀 Key Performance Highlights (100GB Real-World Benchmark)

When restoring a **112.0 GB (100GB-scale)** checkpoint from 5 source shards across 10 target workers:

| Evaluation Dimension | Un-rewritten Baseline | Rewritten Optimized | Gain / Speedup |
| :--- | :--- | :--- | :--- |
| **Restore Wall Time** | **33.35 seconds** | **24.74 seconds** | **1.35x Speedup (-25.8% time saved)** |
| **Effective GCS Throughput** | **3,438.46 MB/s** | **4,635.83 MB/s** | **+1,197.37 MB/s (+1.20 GB/s)** |
| **Time Saved per Restore** | Baseline | **8.61 s per restore** | Eliminates GPU/TPU stall |
| **112 GB Offline CPU Rewrite Time** | N/A | **63.14 seconds** | Low-cost one-time CPU operation |
| **Rewriter Peak Host RAM** | N/A | **< 1.2 GB** | **Zero OOM Risk** (64MB streaming pipeline) |

> 📊 For full empirical data, charts, and mathematical derivations, see [100GB Checkpoint Resharding & Restore Benchmark Report](../../docs/orbax/results/100gb_restore_acceleration.md).

---

## 🛠️ CLI Standalone Rewriter Tool

The repository provides the high-throughput offline rewriter [`tools/checkpoints/orbax_reshard_rewriter.py`](../../tools/checkpoints/orbax_reshard_rewriter.py).

### Usage Scenarios:

#### 1. Cluster Scale-Up Topology Alignment (e.g. 100 TPU $\to$ 500 TPU)
```bash
python3 tools/checkpoints/orbax_reshard_rewriter.py \
  --src-dir="/gcs/my-bucket/checkpoints_100tpu/step_10000" \
  --dst-dir="/gcs/my-bucket/checkpoints_500tpu/step_10000" \
  --strategy=dim_partitions \
  --dim-partitions="0:500" \
  --num-workers=16
```

#### 2. Multi-Dimensional Device Mesh Alignment (FSDP=250 $\times$ TP=2)
```bash
python3 tools/checkpoints/orbax_reshard_rewriter.py \
  --src-dir="/gcs/my-bucket/checkpoints_100tpu/step_10000" \
  --dst-dir="/gcs/my-bucket/checkpoints_500tpu_mesh/step_10000" \
  --strategy=dim_partitions \
  --dim-partitions="0:250,1:2" \
  --num-workers=16
```

#### 3. General Storage I/O Optimization (Merge Small Shards into 64MB Blocks)
```bash
python3 tools/checkpoints/orbax_reshard_rewriter.py \
  --src-dir="/gcs/my-bucket/checkpoints_raw/0" \
  --dst-dir="/gcs/my-bucket/checkpoints_64mb/0" \
  --strategy=optimal_size \
  --target-chunk-mb=64.0 \
  --num-workers=16
```

#### 4. Inference Export: Strip Optimizer States + Bfloat16 Cast
```bash
python3 tools/checkpoints/orbax_reshard_rewriter.py \
  --src-dir="/gcs/my-bucket/training_checkpoint" \
  --dst-dir="/gcs/my-bucket/serving_checkpoint_bf16" \
  --strategy=dim_partitions \
  --dim-partitions="0:8" \
  --strip-opt-state \
  --cast-dtype=bfloat16 \
  --verify \
  --num-workers=16
```

#### 5. Dry-Run Inspection
```bash
python3 tools/checkpoints/orbax_reshard_rewriter.py \
  --src-dir="/gcs/my-bucket/checkpoints_100tpu" \
  --dst-dir="/gcs/my-bucket/checkpoints_500tpu" \
  --strategy=dim_partitions \
  --dim-partitions="0:500" \
  --dry-run
```

---

## ☸️ Helm Chart Parameters Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `gcsfuse.checkpointBucket` | string | `""` | Target GCS bucket for checkpoint storage. |
| `gcsfuse.mountOptions` | string | `"implicit-dirs,file-cache:max-size-mb:-1..."` | GCSFuse mount tuning flags. |
| `workload.srcShards` | int | `5` | Source shard partitioning count along Dim 0. |
| `workload.dstWorkers` | int | `10` | Number of concurrent target restore worker processes. |
| `workload.numLayers` | int | `16` | Number of Transformer model layers (7 matrices / layer). |
| `workload.hiddenDim` | int | `16384` | Hidden dimension size (produces $[16384, 16384]$ Float32 arrays = 1.0 GB / matrix). |
| `workload.numRuns` | int | `3` | Number of timed restore iterations to calculate statistical medians. |
| `workload.numWorkers` | int | `16` | Thread count for parallel offline resharding. |
| `workload.verify` | bool | `true` | Enables post-rewrite numerical precision verification. |
| `nodeSelector` | map | `{"cloud.google.com/gke-nodepool": "n4-standard-80"}` | Kubernetes node selector for compute pods. |

---

## 🚀 Quickstart Deployment

```bash
# 1. Deploy 100GB Orbax Benchmark Release
helm install orbax-bench-100g workloads/orbax-checkpoint-benchmark/helm_chart \
  --set nodeSelector."cloud\.google\.com/gke-nodepool"=n4-standard-80 \
  --set gcsfuse.checkpointBucket="<YOUR_GCS_BUCKET>" \
  --set workload.srcShards=5 \
  --set workload.dstWorkers=10 \
  --set workload.numLayers=16 \
  --set workload.hiddenDim=16384 \
  --set workload.numRuns=3 \
  --set workload.numWorkers=16

# 2. View Live Progress
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=orbax-bench-100g -w

# 3. View Results Log
kubectl logs -l jobset.sigs.k8s.io/jobset-name=orbax-bench-100g -c workload -f
```

---

## 🤖 AI Agent Prompt Examples

You can prompt your AI Agent in natural language to run or analyze Orbax benchmarks:

- *"Run the 100GB Orbax checkpoint benchmark on GKE comparing 5-shard un-rewritten restore vs 10-worker rewritten restore."*
- *"Reshard the checkpoint at gs://my-bucket/ckpt to a 500-shard layout with 16 threads and verify numerical parity."*
- *"Export an inference checkpoint by stripping optimizer states and converting float32 to bfloat16."*

---

## 📚 Complete Documentation Suite

- [Orbax Architectural Overview & Read Storm Analysis](../../docs/orbax/README.md)
- [Step-by-Step Reproduction Guide](../../docs/orbax/step_by_step_guide.md)
- [100GB Checkpoint Resharding & Restore Benchmark Report](../../docs/orbax/results/100gb_restore_acceleration.md)
