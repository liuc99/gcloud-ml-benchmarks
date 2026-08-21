# TensorStore + GCSFuse Distributed Array I/O Benchmark (`tensorstore-gcsfuse`)

A distributed machine learning workload for evaluating large-scale **TensorStore & Zarr** multi-dimensional array read/write performance across Google Cloud Storage (GCSFuse) and Google Kubernetes Engine (GKE).

---

## 🚀 Key Performance Highlights

| Tuning Dimension | Baseline Configuration | Optimized Configuration | Measured Performance Gain |
| :--- | :--- | :--- | :--- |
| **Cluster Scaling (1 to 32 Nodes)** | 1 Node (7.49 GB/s Read) | **32 Nodes (128 Workers)** | **1.35 Tbps (172.9 GB/s) Read**, **863 Gbps Write** |
| **Network MTU Tuning** | Standard 1500 MTU | **8896 Jumbo Frames** | **~83% TCP interrupt reduction**, 7.49 GB/s line rate |
| **Client Protocols** | gRPC (116.8 GB/s Read) | **HTTP/1.1 (Parallel Sockets)** | **+22.3% faster reads (142.8 GB/s)** |
| **Memory Block Buffering** | Default capped buffers | `write:global-max-blocks:-1` | **+107% write speedup** (doubled write throughput) |
| **Chunk Size Slicing** | 50 MB (Metadata penalty) | **200 MB Sweet Spot** | 0.3376s slice retrieval latency |

> 📊 For the complete experimental reports across all 7 dimensions, see the [TensorStore Results Suite](../../docs/tensorstore/README.md#benchmark-results-by-tuning-dimension).

---

## ☸️ Helm Chart Parameters Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `workload.nodes` | int | `1` | Number of distributed compute nodes. |
| `workload.numWorkers` | int | `8` | Worker processes per node. |
| `workload.tensorstoreShape` | string | `"1000,1000,100"` | Global array shape dimensions. |
| `workload.tensorstoreChunks` | string | `"100,100,100"` | Zarr chunk shape geometry. |
| `workload.tensorstoreDtype` | string | `"float32"` | Array data type (`float32`, `bfloat16`, `int32`). |
| `workload.tensorstoreDriver` | string | `"zarr"` | TensorStore driver backend (`zarr`, `n5`). |
| `gcsfuse.enabled` | bool | `true` | Enables GCSFuse CSI Driver sidecar mount. |
| `gcsfuse.datasetBucket` | string | `""` | Target GCS bucket for array reading. |
| `gcsfuse.checkpointBucket` | string | `""` | Target GCS bucket for array writing. |
| `gcsfuse.mountOptions` | string | `"implicit-dirs"` | GCSFuse mount options (e.g. `client-protocol:http1`). |
| `nodeSelector` | map | `{"cloud.google.com/gke-nodepool": "c4-standard-192"}` | Target GKE node pool selector. |

---

## 🚀 Quickstart Deployment

```bash
# Deploy TensorStore multi-node benchmark release
helm install tensorstore-bench workloads/tensorstore-gcsfuse/helm_chart -f workloads/tensorstore-gcsfuse/helm_chart/values_base.yaml \
  --set workload.nodes=4 \
  --set workload.numWorkers=8 \
  --set gcsfuse.checkpointBucket="<YOUR_BUCKET>" \
  --set-string gcsfuse.mountOptions="implicit-dirs\,write:global-max-blocks:-1\,file-cache:max-size-mb:-1"
```

---

## 📚 Complete Documentation Suite

- [TensorStore Documentation Index](../../docs/tensorstore/README.md)
- [Step-by-Step Reproduction Guide](../../docs/tensorstore/step_by_step_guide.md)
- [Multi-Node Cluster Scaling (1 to 32 Nodes)](../../docs/tensorstore/results/node_scaling.md)
- [Network MTU Tuning (8896 Jumbo Frames vs 1500 MTU)](../../docs/tensorstore/results/network_mtu.md)
- [Client Protocols (HTTP/1.1 vs gRPC)](../../docs/tensorstore/results/client_protocols.md)
- [Zarr Chunk Size & Slicing Latency](../../docs/tensorstore/results/chunk_size_and_file_size.md)
- [GCSFuse Memory Block Buffer Tuning](../../docs/tensorstore/results/global_max_blocks.md)
- [Worker Process Concurrency](../../docs/tensorstore/results/process_concurrency.md)
- [Thread Concurrency & I/O Parallelism](../../docs/tensorstore/results/thread_concurrency.md)
