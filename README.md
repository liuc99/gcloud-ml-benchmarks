# gcloud-ml-benchmarks

A unified benchmarking suite for evaluating high-throughput Machine Learning I/O performance on Google Cloud Platform (GCP) and Google Kubernetes Engine (GKE).

This repository measures dataset loading, array I/O, and model checkpointing performance across various storage backends, client protocols, and file system mounts.

---

## 🎯 Supported Workloads & Storage Backends

| Workload | Backend / Storage Mount | Description | Key Metrics Evaluated |
| :--- | :--- | :--- | :--- |
| **`tensorstore-gcsfuse`** | GCSFuse CSI Driver (Zarr / TensorStore) | Multi-node distributed read & write benchmarks of multi-dimensional Zarr arrays over GCSFuse (and TensorStore drivers). | Cluster Read/Write Throughput (up to 1.35 Tbps), MTU 8896 vs 1500 tuning, HTTP/1.1 vs gRPC protocol impact. |
| **`hf-pytorch-lightning-cpu`** (Lustre) | Google Cloud Managed Lustre (`LustreCsiDriver`) | PyTorch DDP model training & checkpointing over high-performance Managed Lustre Parallel File System. | Restore I/O speed, Checkpoint save time, PyTorch DDL throughput on Lustre. |
| **`hf-pytorch-lightning-cpu`** (GCSFuse) | GCSFuse CSI Driver (`GcsFuseCsiDriver`) | PyTorch DDP training & checkpointing reading/writing directly via GCSFuse sidecar mounts. | Streaming write throughput, checkpoint saving, dataset loading over GCS. |
| **`hf-pytorch-lightning-cpu`** (GCSFS) | FSSpec / GCSFS Python Client | Native Python `gcsfs` file system interface for PyTorch checkpointing and dataset staging. | `gcsfs` chunk size efficiency, Python GIL impact, raw GCS HTTP API latency. |

---

## 📁 Repository & Documentation Structure

```
gcloud-ml-benchmarks/
├── cloudbuild/                  # CloudBuild automation & GCP trigger definitions
│   ├── macrobenchmarks-tensorstore-gcsfuse-cloudbuild.yaml
│   ├── macrobenchmarks-gcsfuse-cloudbuild.yaml  # Runs GCSFuse & Managed Lustre tests
│   ├── macrobenchmarks-cloudbuild.yaml          # Runs gcsfs tests
│   ├── macrobenchmarks-ingestion-cloudbuild.yaml
│   └── scripts/                 # Helm, GKE cluster provisioning, Lustre PVC setup scripts
├── workloads/                   # Benchmark workload definitions & Helm charts
│   ├── tensorstore-gcsfuse/     # TensorStore multi-node array I/O harness
│   │   └── helm_chart/
│   └── hf-pytorch-lightning-cpu/ # PyTorch Llama 3.1 8B DDP training & checkpoint harness
│       └── helm_chart/
└── docs/                        # Documentation suite organized by benchmark workload
    ├── README.md                # Master documentation index
    ├── tensorstore/             # TensorStore + GCSFuse benchmark suite
    │   ├── step_by_step_guide.md# Manual reproduction guide
    │   └── results/             # Results by dimension (scaling, MTU, protocols, chunks, concurrency)
    └── pytorch/                 # PyTorch + Storage benchmark suite
        ├── step_by_step_guide.md# Manual reproduction guide (Lustre, GCSFuse, gcsfs)
        └── results/             # PyTorch benchmark results
```

---

## 🚀 Quick Links & Documentation

* 📦 **TensorStore Benchmarks**:
  * [**TensorStore Documentation Index**](docs/tensorstore/README.md)
  * [**Step-by-Step Reproduction Guide**](docs/tensorstore/step_by_step_guide.md)
  * [**Multi-Node Cluster Scaling (1 to 32 Nodes)**](docs/tensorstore/results/node_scaling.md)
  * [**Network MTU Tuning (8896 Jumbo Frames vs 1500 MTU)**](docs/tensorstore/results/network_mtu.md)
  * [**Client Protocols (HTTP/1.1 vs gRPC)**](docs/tensorstore/results/client_protocols.md)
  * [**Zarr Chunk Size & Slicing Latency**](docs/tensorstore/results/chunk_size_and_file_size.md)
  * [**Worker Concurrency & Mount Tuning**](docs/tensorstore/results/concurrency_and_mount_tuning.md)

* 🔥 **PyTorch Benchmarks**:
  * [**PyTorch Documentation Index**](docs/pytorch/README.md)
  * [**Step-by-Step Reproduction Guide (Managed Lustre, GCSFuse, gcsfs)**](docs/pytorch/step_by_step_guide.md)

---

## ⚡ CloudBuild Execution Commands

### 1. TensorStore + GCSFuse Benchmark
```bash
gcloud builds submit \
  --config=cloudbuild/macrobenchmarks-tensorstore-gcsfuse-cloudbuild.yaml \
  --substitutions=_PROJECT_ID="your-project-id",_ZONE="us-central1-b",_NODES="32",_MACHINE_TYPE="n4-standard-80"
```

### 2. PyTorch + Google Cloud Managed Lustre Benchmark
```bash
gcloud builds submit \
  --config=cloudbuild/macrobenchmarks-gcsfuse-cloudbuild.yaml \
  --substitutions=_USE_LUSTRE="true",_LUSTRE_INSTANCE="your-lustre-instance-id",_NODES="2",_RANKS_PER_NODE="4"
```

### 3. PyTorch + GCSFuse Benchmark
```bash
gcloud builds submit \
  --config=cloudbuild/macrobenchmarks-gcsfuse-cloudbuild.yaml \
  --substitutions=_USE_GCSFUSE="true",_GCSFUSE_ENABLE_STREAM_WRITE="true",_NODES="2",_RANKS_PER_NODE="4"
```

---

## 📊 Metrics & Data Ingestion

Benchmark runs collect node-level and cluster-level performance metrics (throughput MB/s, checkpoint duration, peak RSS memory, GCSFuse metrics). Results can be automatically ingested into BigQuery using `cloudbuild/macrobenchmarks-ingestion-cloudbuild.yaml` and the schema defined in `cloudbuild/macrobenchmarks_schema.json`.
