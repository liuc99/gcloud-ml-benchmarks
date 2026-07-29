# Documentation & Benchmark Results Index

This directory contains the organized step-by-step guides and detailed performance results across all benchmark dimensions.

---

## 🛠️ Step-by-Step Reproduction Guide

* [**Step-by-Step Reproduction Guide**](step_by_step_guide.md): Complete guide for setting up GKE clusters, Zonal RAPID buckets with HNS, Workload Identity, Helm charts, and running benchmarks.

---

## 📊 Performance Benchmark Results by Dimension

The benchmark results are categorized by individual tuning dimensions in the [`results/`](results/) directory:

1. [**Multi-Node Cluster Scaling**](results/node_scaling.md)
   * **Dimension**: Number of nodes (1 to 32 nodes / 8 to 128 worker ranks / 3.81 TB data).
   * **Highlights**: Terabit-scale aggregate throughput (**1.35 Tbps / 172.9 GB/s Read**, **863 Gbps / 107.8 GB/s Write**).

2. [**Network MTU Tuning**](results/network_mtu.md)
   * **Dimension**: 8896 Jumbo Frames vs 1500 Standard MTU.
   * **Highlights**: ~83% reduction in TCP packet interrupts, single-node peak read reaching **7.49 GB/s** (saturating 50 Gbps NIC).

3. [**Client Protocol Selection**](results/client_protocols.md)
   * **Dimension**: `client-protocol=http1` (HTTP/1.1) vs `client-protocol=grpc` (gRPC).
   * **Highlights**: HTTP/1.1 delivers **+22.3% faster reads** (142.8 GB/s) via parallel sockets; gRPC delivers **+13.6% faster writes** (107.8 GB/s) via HTTP/2 streaming.

4. [**Zarr Chunk Size & Slicing Latency**](results/chunk_size_and_file_size.md)
   * **Dimension**: 50 MB vs 200 MB vs 400 MB Zarr chunk sizes.
   * **Highlights**: 200 MB chunk sweet spot (**0.3376s** slice retrieval latency), 50 MB chunk metadata overhead (-80% write penalty).

5. [**Worker Concurrency & Mount Tuning**](results/concurrency_and_mount_tuning.md)
   * **Dimension**: 1 vs 4 vs 8 worker processes per node, `write:global-max-blocks:-1` memory block un-capping.
   * **Highlights**: Doubled write throughput (**+107% speedup**) by un-capping memory block allocation.
