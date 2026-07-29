# TensorStore + GCSFuse Benchmark Suite

This directory contains the reproduction guide and detailed experimental performance results for the **TensorStore + GCSFuse** workload.

---

## 📁 Directory Structure

```
docs/tensorstore/
├── README.md                           # TensorStore workload overview & index
├── step_by_step_guide.md               # End-to-end manual GKE reproduction guide
└── results/                            # Test results organized by benchmark dimension
    ├── node_scaling.md                 # 1 to 32 nodes (up to 1.35 Tbps aggregate read)
    ├── network_mtu.md                  # 8896 Jumbo Frames vs 1500 MTU
    ├── client_protocols.md             # HTTP/1.1 vs gRPC protocol comparison
    ├── chunk_size_and_file_size.md     # 50MB vs 200MB vs 400MB chunk size & slice retrieval
    └── concurrency_and_mount_tuning.md # Worker scaling & global-max-blocks tuning
```

---

## 🛠️ Step-by-Step Reproduction Guide

* [**Step-by-Step Reproduction Guide**](step_by_step_guide.md): Complete setup instructions for creating Zonal RAPID buckets, GKE clusters with 8896 MTU, Workload Identity, Helm charts, and running TensorStore array benchmarks.

---

## 📊 Benchmark Results by Tuning Dimension

1. [**Multi-Node Cluster Scaling**](results/node_scaling.md)
   * **Dimension**: 1 to 32 nodes (128 worker ranks / 3.81 TB dataset).
   * **Highlights**: Terabit-scale aggregate throughput (**1.35 Tbps / 172.9 GB/s Read**, **863 Gbps / 107.8 GB/s Write**).

2. [**Network MTU Tuning**](results/network_mtu.md)
   * **Dimension**: 8896 Jumbo Frames vs 1500 Standard MTU.
   * **Highlights**: ~83% reduction in TCP packet interrupts, single-node peak read reaching **7.49 GB/s**.

3. [**Client Protocol Selection**](results/client_protocols.md)
   * **Dimension**: `client-protocol=http1` (HTTP/1.1) vs `client-protocol=grpc` (gRPC).
   * **Highlights**: HTTP/1.1 delivers **+22.3% faster reads** (142.8 GB/s) via parallel sockets; gRPC delivers **+13.6% faster writes** (107.8 GB/s) via HTTP/2 streaming.

4. [**Zarr Chunk Size & Slicing Latency**](results/chunk_size_and_file_size.md)
   * **Dimension**: 50 MB vs 200 MB vs 400 MB Zarr chunk sizes.
   * **Highlights**: 200 MB chunk sweet spot (**0.3376s** slice retrieval latency), 50 MB chunk metadata overhead (-80% write penalty).

5. [**Worker Concurrency & Mount Tuning**](results/concurrency_and_mount_tuning.md)
   * **Dimension**: 1 vs 4 vs 8 worker processes per node, `write:global-max-blocks:-1` memory block un-capping.
   * **Highlights**: Doubled write throughput (**+107% speedup**) by un-capping memory block allocation.
