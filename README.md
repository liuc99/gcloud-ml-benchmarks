# gcloud-ml-benchmarks: AI Agent-Driven ML Storage Benchmark & PoC Suite

A unified, AI Agent-driven benchmarking and PoC harness for evaluating high-throughput Machine Learning I/O performance on Google Cloud Platform (GCP) and Google Kubernetes Engine (GKE).

This open-source suite empowers GCP ML developers, data engineers, and AI Agent assistants to interactively trigger real-world ML training Demos, PoCs, and I/O performance benchmarks across **Google Cloud Storage (GCSFuse & `gcsfs`)** and **Google Cloud Managed Lustre**.

---

## 🎯 Supported Storage Systems & Workloads

### 💾 Primary Storage Systems:
1. **Google Cloud Storage (GCS)**: Evaluated via **GCSFuse Streaming Writes** (`GcsFuseCsiDriver`) and native Python **`gcsfs` REST API**.
2. **Google Cloud Managed Lustre**: High-performance parallel file system evaluated via `LustreCsiDriver`.

### 🚀 Workload Harnesses:
- **PyTorch DDP (`hf-pytorch-lightning-cpu`)**: Simulates Llama 3.1 8B multi-node distributed training, dataset streaming, and 45 GB model state dict checkpointing.
- **TensorStore / Zarr (`tensorstore-gcsfuse`)**: Multi-node array read/write benchmark evaluating chunking, concurrency, network MTU tuning, and gRPC streaming.
- *(Extensible)* Architecture designed for easily plugging in future ML frameworks (e.g. JAX, MaxText, Ray, vLLM).

---

## 🤖 AI Agent Automated Benchmarking (Modular AI Agent Skills)

This repository includes a suite of modular, cross-platform **AI Agent Skills** (located in `skills/` and `.gemini/skills/`):

- 🎯 **`ml-benchmark-orchestrator`**: Master orchestrator. **Interactively interviews the user first** to clarify workload, storage backend, resource preferences, iterations, and custom flags before taking any action.
- ☁️ **`gcp-resource-provisioner`**: Auto-discovers or provisions GKE clusters, Managed Lustre PVCs, GCS buckets, and Workload Identity IAM bindings.
- 🚀 **`helm-workload-runner`**: Dynamically constructs Helm commands, deploys benchmark releases, monitors Pod execution asynchronously, and tears down releases.
- 📊 **`benchmark-metrics-parser`**: Parses container stdout logs via `parse_metrics.py` to extract MB/s throughput, latency, and duration metrics into Markdown reports.

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
├── skills/                      # Modular AI Agent Skills for benchmark automation
│   ├── ml-benchmark-orchestrator/ # Master orchestrator & interactive user interview
│   ├── gcp-resource-provisioner/ # GKE, Lustre PVC, GCS bucket & IAM setup
│   ├── helm-workload-runner/   # Dynamic Helm execution, async tracking & teardown
│   └── benchmark-metrics-parser/# Empirical log parsing & Markdown report generator
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
  * [**GCSFuse Memory Block Buffer Tuning**](docs/tensorstore/results/global_max_blocks.md)
  * [**Worker Process Concurrency**](docs/tensorstore/results/process_concurrency.md)
  * [**Thread Concurrency & I/O Parallelism**](docs/tensorstore/results/thread_concurrency.md)

* 🔥 **PyTorch Benchmarks**:
  * [**PyTorch Documentation Index**](docs/pytorch/README.md)
  * [**Step-by-Step Reproduction Guide (Managed Lustre, GCSFuse, gcsfs)**](docs/pytorch/step_by_step_guide.md)
  * [**Storage Backends Comparison (Lustre vs GCSFuse vs gcsfs)**](docs/pytorch/results/storage_backends.md)
  * [**Model Checkpointing Performance & Streaming**](docs/pytorch/results/checkpoint_performance.md)
  * [**Rank Topology & RAM OOM Prevention**](docs/pytorch/results/rank_scaling_and_memory.md)

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
