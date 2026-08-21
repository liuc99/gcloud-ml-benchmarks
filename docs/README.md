# Documentation & Benchmark Suite Index

This directory contains the workload-specific reproduction guides, architectural documentation, and empirical benchmark results for `gcloud-ml-benchmarks`.

---

## 📚 Workload Documentation Directories

| Workload | Directory | Description & Included Guides |
| :--- | :--- | :--- |
| **Orbax & TensorStore Checkpoints** | [`docs/orbax/`](orbax/) | Checkpoint offline resharding, topology adaptation (e.g. 5 shards $\to$ 10 workers or 100 $\to$ 500 TPU), optimizer state stripping, and 100GB restore acceleration over GCSFuse / Zonal RAPID GCS. |
| **MaxText Dataset Loader** | [`docs/maxtext/`](maxtext/) | MaxText JAX LLM training input pipeline benchmark evaluating Parquet Range Reads, ArrayRecord pre-tokenized streaming, and in-tree data loader over GCS and GCSFuse. |
| **Multi-Format Dataset Loader** | [`docs/multi_format_dataset/`](multi_format_dataset/) | Multi-format dataset streaming benchmark harness evaluating Parquet, WebDataset TAR, Zarr/TensorStore, PyTorch `.pt`, and JSONL ingestion throughput, TTFB, and latency. |
| **TensorStore + GCSFuse** | [`docs/tensorstore/`](tensorstore/) | Multi-node distributed Zarr array I/O over GCSFuse. Includes 1-to-32 node scaling (1.35 Tbps), MTU 8896, HTTP/1.1 vs gRPC, chunk size, and worker concurrency results. |
| **PyTorch + Storage** | [`docs/pytorch/`](pytorch/) | PyTorch DDP Llama 3.1 8B training & checkpointing over Google Cloud Managed Lustre (`LustreCsiDriver`), GCSFuse sidecars, and `gcsfs`. |

---

## 🗂️ Detailed Directory Structure

```
docs/
├── README.md                           # Master documentation index
│
├── orbax/                              # Orbax & TensorStore Checkpoint Suite
│   ├── README.md                       # Orbax workload overview & read storm analysis
│   ├── step_by_step_guide.md           # Reproduction guide for GKE & GCS Zonal RAPID
│   └── results/                        # Empirical benchmark results
│       └── 100gb_restore_acceleration.md # 100GB empirical restore, shard layout & rewriter benchmark
│
├── maxtext/                            # MaxText Parquet & ArrayRecord Benchmark Suite
│   ├── README.md                       # MaxText workload overview & architecture
│   ├── parquet_range_reads_guide.md    # Reproduction guide (Native GCS & GCSFuse)
│   └── results/                        # Empirical benchmark results
│       ├── parquet_vs_arrayrecord.md   # 420 GB vs 155 GB footprint & step latency
│       ├── shuffle_strategies.md       # None vs Two-Stage vs Global shuffle latency
│       └── storage_access_modes.md     # GCSFuse CSI mount vs Native GCS client
│
├── multi_format_dataset/               # Multi-Format Dataset Loading Suite
│   ├── README.md                       # Workload overview & format matrix
│   ├── step_by_step_guide.md           # Reproduction guide for GCSFuse, gcsfs & Lustre
│   └── results/                        # Empirical benchmark results
│       ├── arrayrecord_range_read_chunk_size_scaling.md    # ArrayRecord Range Read Chunk Size Scaling on GCSFuse CSI
│       ├── hf_parquet_manifest_comparison.md               # HF Parquet Manifest.json vs Dynamic Globbing
│       ├── gcsfuse_vs_direct_gcs_parquet.md                # GCSFuse CSI Mount vs Direct GCS (gcsfs)
│       └── format_comparison.md                            # Format comparison (Parquet vs TAR vs Zarr vs PT)
│
├── tensorstore/                        # TensorStore + GCSFuse Benchmark Suite
│   ├── README.md                       # TensorStore documentation index
│   ├── step_by_step_guide.md           # Reproduction guide for TensorStore GKE benchmarks
│   └── results/                        # Benchmark results by dimension
│       ├── node_scaling.md             # 1 to 32 nodes (up to 1.35 Tbps aggregate read)
│       ├── network_mtu.md              # 8896 Jumbo Frames vs 1500 MTU
│       ├── client_protocols.md         # HTTP/1.1 vs gRPC protocol comparison
│       ├── chunk_size_and_file_size.md # 50MB vs 200MB vs 400MB chunk size & slice retrieval
│       ├── global_max_blocks.md        # GCSFuse memory block buffer tuning (write:global-max-blocks)
│       ├── process_concurrency.md      # Worker process concurrency scaling (1 vs 4 vs 8 processes)
│       └── thread_concurrency.md       # Application I/O thread concurrency scaling
│
└── pytorch/                            # PyTorch + Storage Benchmark Suite
    ├── README.md                       # PyTorch documentation index
    ├── step_by_step_guide.md           # Reproduction guide for Managed Lustre, GCSFuse & gcsfs
    └── results/                        # Empirical benchmark results
        ├── checkpoint_write_performance.md # Checkpoint pickling & network upload throughput (Decoupled write)
        ├── checkpoint_restore_performance.md # Checkpoint restore speed & kernel VFS caching (Decoupled restore)
        └── rank_scaling_and_memory.md  # Rank topology & RAM OOM prevention
```
