# Benchmark Dimension: PyTorch Model Checkpoint Performance

This document provides a detailed breakdown of **PyTorch DDP Checkpointing Mechanics**, analyzing in-memory state dict serialization (CPU pickling), network upload throughput, and total elapsed duration for **45 GB (44.87 GiB)** model & optimizer checkpoints saved across different storage backends.

---

## 📊 Checkpoint Save Duration Breakdown

Evaluation of a 44.87 GiB Llama 3.1 8B checkpoint (including model weights, AdamW optimizer states, and trainer state) saved at 25-step intervals during 100-step DDP training runs:

| Storage Location | Pickling & Serialization Time | Network Upload Time | Total Elapsed Save Time | Network Write Rate | Aggregate Save Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Managed Lustre (`/lustre/checkpoints`)** | **5.01s** | **48.15s** | **53.16 seconds** | **953.41 MB/s** | **863.70 MB/s** |
| **GCSFuse (`/gcs/checkpoints`)** | **5.00s** | **75.14s** | **80.14s – 96.30s** | **611.51 MB/s** | **503.29 MB/s** |
| **Direct GCS (`gs://.../checkpoints_gcsfs`)** | **5.00s** | **233.53s** | **238.53 seconds** | **~550.00 MB/s** | **192.64 MB/s** |

---

## 🔍 Detailed Analysis of Checkpointing Phasing

### Phase 1: In-Memory Pickling & Serialization (~5.0s)
- During the `on_train_epoch_end` / `every_n_train_steps` callback, PyTorch Lightning rank 0 gathers the state dictionary and serializes it via CPU `torch.save()` / `pickle`.
- In-memory pickling duration remains constant at **~5.0 seconds** regardless of target storage backend.

### Phase 2: Storage I/O & Network Streaming
- **Managed Lustre (53.16s Total)**: The parallel POSIX client streams data blocks directly to Lustre Object Storage Targets (OSTs) over GCP VPC internal network at **953.41 MB/s**.
- **GCSFuse Streaming (80.14s Total)**: Streaming write buffers push HTTP/2 gRPC data blocks directly to GCS at **611.51 MB/s**, avoiding local disk writes.
- **Direct GCS `gcsfs` (238.53s Total)**: Python `fsspec` handles buffer uploads in chunked HTTP POST requests, introducing user-space thread locks that reduce overall save speed to **192.64 MB/s**.

---

## 💡 Recommendations for Large-Scale PyTorch Checkpointing

1. **Prefer Managed Lustre for Frequent Checkpoints**: When model checkpointing occurs frequently (e.g. every 25–50 steps), Managed Lustre minimizes GPU/CPU idle time during saves by **77.7%**.
2. **Enable GCSFuse Streaming Writes**: If saving directly to Object Storage, always configure `write:enable-streaming-writes:true` and `write:global-max-blocks:-1` to sustain **>600 MB/s** network write rates.
3. **Dedicated Writer Rank Topology**: Configure PyTorch Lightning `LoggedModelCheckpoint` so only Rank 0 writes checkpoint bytes while non-writer ranks skip upload, eliminating redundant I/O contention.
