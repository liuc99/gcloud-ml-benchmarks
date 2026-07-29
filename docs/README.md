# Documentation & Benchmark Suite Index

This directory contains the workload-specific reproduction guides and experimental benchmark results for `gcloud-ml-benchmarks`.

---

## 🎯 Workload Documentation Directories

| Workload | Directory | Description & Included Guides |
| :--- | :--- | :--- |
| **TensorStore + GCSFuse** | [`docs/tensorstore/`](tensorstore/) | Multi-node distributed Zarr array I/O over GCSFuse. Includes 1-to-32 node scaling (1.35 Tbps), MTU 8896, HTTP/1.1 vs gRPC, chunk size, and worker concurrency results. |
| **PyTorch + Storage** | [`docs/pytorch/`](pytorch/) | PyTorch DDP Llama 3.1 8B training & checkpointing over Google Cloud Managed Lustre (`LustreCsiDriver`), GCSFuse sidecars, and `gcsfs`. |

---

## 📁 Detailed Directory Structure

```
docs/
├── README.md                           # Master documentation index
├── tensorstore/                        # TensorStore + GCSFuse benchmark suite
│   ├── README.md                       # TensorStore documentation index
│   ├── step_by_step_guide.md           # Reproduction guide for TensorStore GKE benchmarks
│   └── results/                        # Dimension results
│       ├── node_scaling.md             # 1 to 32 nodes (up to 1.35 Tbps aggregate read)
│       ├── network_mtu.md              # 8896 Jumbo Frames vs 1500 MTU
│       ├── client_protocols.md         # HTTP/1.1 vs gRPC protocol comparison
│       ├── chunk_size_and_file_size.md # 50MB vs 200MB vs 400MB chunk size & slice retrieval
│       ├── global_max_blocks.md        # GCSFuse memory block buffer tuning (write:global-max-blocks)
│       ├── process_concurrency.md      # Worker process concurrency scaling (1 vs 4 vs 8 processes)
│       └── thread_concurrency.md       # Application I/O thread concurrency scaling
│
└── pytorch/                            # PyTorch + Storage benchmark suite
    ├── README.md                       # PyTorch documentation index
    ├── step_by_step_guide.md           # Reproduction guide for Managed Lustre, GCSFuse & gcsfs
    └── results/                        # Test results across storage backends
        ├── storage_backends.md         # Managed Lustre vs GCSFuse vs Direct GCS (gcsfs)
        ├── checkpoint_performance.md   # Checkpoint pickling & network upload throughput
        └── rank_scaling_and_memory.md  # Rank topology & RAM OOM prevention
```
