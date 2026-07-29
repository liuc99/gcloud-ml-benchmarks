# Benchmark Dimension: Storage Backends (Managed Lustre vs GCSFuse vs Direct GCS / `gcsfs`)

This document presents a head-to-head performance comparison of **Google Cloud Managed Lustre**, **GCSFuse Streaming Writes**, and **Direct GCS (`gcsfs`)** when executing PyTorch Distributed Data Parallel (DDP) model training (Llama 3.1 8B) and 45 GB model checkpointing on Google Kubernetes Engine (GKE).

---

## 📊 Summary Performance Comparison Table

Single-node 2-rank DDP evaluation on `n4-standard-80` (80 vCPUs, 320 GB RAM, 50 Gbps NIC) performing 100 training steps and saving **45 GB (44.87 GiB)** PyTorch model & optimizer state dict checkpoints:

| Storage Backend | Interface & Protocol | Dataset Load Latency | Raw Checkpoint Write Speed | Aggregate Save Duration (45 GB) | Aggregate Checkpoint Throughput | Overall Performance & Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Google Cloud Managed Lustre** | POSIX (`LustreCsiDriver`) | **< 0.10s** | **953.41 MB/s (~9.53 Gbps)** | **53.16 seconds** | **863.70 MB/s** | **FASTEST (~4.5x faster save than GCS)**; Zero-overhead parallel POSIX file system streaming. |
| **GCSFuse (Streaming Writes)** | FUSE Mount (`implicit-dirs`) | **0.82s** | **611.51 MB/s (~6.12 Gbps)** | **75.14s – 96.30s** | **503.29 MB/s** | **OPTIMAL OBJECT STORAGE**; Streamed write buffers eliminate local disk staging. |
| **Direct GCS (`gcsfs`)** | Python `fsspec` REST API | **1.52s** | **~550.00 MB/s** | **238.53 seconds** | **192.64 MB/s** | Single-stream REST API serialization overhead causes write backpressure. |

---

## 🔍 Key Technical Findings

### 1. Managed Lustre Delivers ~4.5x Checkpointing Speedup
- **Parallel POSIX Write Speed**: Managed Lustre achieved **953.41 MB/s** raw write throughput to the `/lustre/checkpoints` volume.
- **Save Time Reduction**: Saving the full 44.87 GiB Llama 3.1 8B checkpoint completed in just **53.16 seconds** on Lustre, compared to 238.53 seconds on `gcsfs`—reducing training stall times by **77.7%**.

### 2. GCSFuse Streaming Writes Outperform `gcsfs` by 2.6x
- Enabling `write:enable-streaming-writes:true` on GCSFuse achieved **611.51 MB/s** network write throughput directly to Google Cloud Storage buckets.
- GCSFuse streaming bypasses local ephemeral disk capacity limits and avoids the single-threaded REST API serialization bottlenecks seen in `gcsfs`.

### 3. Fast HuggingFace Dataset Ingestion
- HuggingFace dataset initialization completed in **0.82 seconds** over GCSFuse and **1.52 seconds** over `gcsfs`.
- Both object storage integration methods provided instant data loading without requiring pre-staging datasets onto local SSDs.

---

## ⚙️ Configuration Summary

```yaml
# Managed Lustre PVC Configuration
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: lustre-checkpoint-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 12000Gi
  storageClassName: ""
```

```bash
# GCSFuse Mount Options for CloudBuild / Helm
_GCSFUSE_MOUNT_OPTIONS="implicit-dirs,client-protocol=grpc,write:enable-streaming-writes:true,write:global-max-blocks:-1"
```
