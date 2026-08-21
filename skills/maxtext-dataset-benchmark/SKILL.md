---
name: maxtext-dataset-benchmark
description: MANDATORY skill for all MaxText dataset loading benchmarks and evaluations. MUST be viewed and referenced FIRST whenever the user asks to benchmark or test MaxText dataset loading, Parquet GCS range reads, ArrayRecord pre-tokenized streaming, GCSFuse mount options (including StorageClass Profiles like gcsfusecsi-training), Rapid (zonal) vs Regional GCS bucket comparisons, dataset manifest generation, or MaxText shuffle strategies (none, two_stage, global).
---

# MaxText Dataset Benchmark & Conversion Sub-Skill (`maxtext-dataset-benchmark`)

This is a workload sub-skill for executing MaxText dataset loading benchmarks on GKE. It handles the technical execution of **Parquet GCS Range Reads**, **ArrayRecord pre-tokenized streaming**, the **`parquet_to_arrayrecord.py` multi-process converter**, **`manifest.json` metadata index generation**, **GKE StorageClass Profiles (`gcsfusecsi-training`, `gcsfusecsi-checkpointing`, `gcsfusecsi-serving`)**, **Regional vs. Rapid (Zonal) GCS Bucket performance comparison**, and **Two-Stage Shuffle (`none`, `two_stage`, `global`)** benchmark runs.

*Note*: Master interactive questionnaires and Plan Approval Protocols are governed by [`ml-benchmark-orchestrator`](../ml-benchmark-orchestrator/SKILL.md).

> [!CAUTION]
> **STRICT PLAN ADHERENCE & MANDATORY PLAN REVIEW PROTOCOL**:
> 1. **MANDATORY PLAN REVIEW FOR ALL RUNS (INITIAL & SUPPLEMENTAL)**: The agent MUST ALWAYS present a structured Execution Plan Review table and wait for explicit user confirmation ("Proceed" / "确认") before deploying Helm releases or executing benchmarks. This rule applies equally to incremental/supplemental requests.
> 2. **NO AD-HOC INLINE CODE**: The agent MUST NOT generate or execute ad-hoc Python snippets (`python3 -c "..."`) or temporary debugging code. All operations MUST use formal CLI tools in `tools/` committed to the repository.
> 3. **FAIL-FAST ON ERRORS**: If any resource provisioning step or tool execution fails or encounters quota/API restrictions, the agent MUST NOT attempt silent fallbacks or write ad-hoc workaround scripts. It MUST immediately pause, report the exact error/Traceback to the user, present the proposed plan amendment, and wait for explicit user re-approval before proceeding.
> 4. **MANDATORY DATASET PARITY & NO SILENT SUBSTITUTION**: In comparative benchmarks (e.g. Regional vs. Rapid bucket, Parquet vs. ArrayRecord), all storage backends MUST use identically configured datasets (same total samples, same sequence length, same token dimensions).
> 5. **MANDATORY BUCKET NAME PARITY GATEWAY**: The agent MUST verify that the resolved bucket name returned by `tools/infrastructure/bucket_manager.py` matches the exact bucket name specified in the user-approved execution plan.
> 6. **STRICT PROHIBITION AGAINST AUTONOMOUS BUCKET SCANNING**: The agent MUST NEVER autonomously scan, query, or enumerate buckets in the GCP project (`gcloud storage ls`, `bucket_manager.py --action=list`, etc.) on its own. All target buckets MUST be explicitly supplied by the user.

---

## 💡 Production Engineering Insights: Parquet vs. ArrayRecord

| 评估维度 | Parquet (未预分词) | ArrayRecord (Pre-tokenized) | 生产推荐与优势说明 |
| :--- | :--- | :--- | :--- |
| **适用阶段** | 离线 ETL、数据湖仓分析 | 大模型分布式预训练、微调与长文本训练 | **ArrayRecord 为训练端黄金标准** |
| **Shuffle 机制支持** | **物理上无法支持 True Global Shuffle**（仅支持 2-Stage 滑动窗口近似） | **原生支持在线 True Global Shuffle**（基于 Record 字节索引表 + Grain 伪随机置换） | **ArrayRecord 消除离线预打散依赖** |
| **存储与传输体积** | 420.10 GB (100%) | 155.27 GB (**节省 63.9%**) | **ArrayRecord 大幅降低跨节点网络拥塞** |
| **训练端 CPU 负载** | **极高（占单步耗时 79% ~ 92%）** | **零计算负载 (Zero-CPU Ingestion)** | **ArrayRecord 彻底消除 GPU 供数饥饿** |
| **单步尾部延迟 (p99)** | 抖动严重（高分位尖峰可达 290 ms） | **极度平稳（p99 收敛在 20~40 ms）** | **ArrayRecord 消除 AllReduce 同步木桶效应** |
| **Step 1 前冷启动 (TTFB)** | 缓慢（需对 20,000 条样本完成 CPU 分词） | **极快（仅纯二进制 I/O 拉取）** | **ArrayRecord 冷启动提速 2~10 倍** |

### 🔀 核心存储架构解析：Parquet 无法进行在线 True Global Shuffle 的物理原因
1. **Row-Group 压缩屏障**：Parquet 数据按 Row Group（通常 5,000 ~ 50,000 行）整块压缩存储。为了读取任意随机的一行数据，PyArrow 必须下载并解压整个 Row Group，导致 **5,000x ~ 50,000x 的解压 CPU 与 I/O 放大**。
2. **2-Stage 为 Parquet 唯一流式可行解**：在线流式读取 Parquet 时，只能先做 **Shard 级全局乱序**，再由多个 Worker 并发顺序解压不同的 Shard，并在内存中维持一个滑动窗口 Buffer（如 10,000 条）进行局部混杂抽样。要实现严格无重复的 True Global Shuffle，Parquet 只能在训练前通过 Spark/Ray 执行耗时的**离线预打散 ETL**。
3. **ArrayRecord 的破局设计**：ArrayRecord 为每条样本维护物理 64-bit 偏移量索引表，配合 Grain（`grain.python.ArrayRecordDataSource`）可直接根据全局伪随机置换索引进行点对点 Range Read，**天然保证单 Epoch 内所有样本 0 重复、0 遗漏且完全真随机**。

---

## ⚙️ GKE GCSFuse StorageClass Profiles

GKE supports automated performance tuning via built-in StorageClass Profiles (`kubectl get sc -l gke-gcsfuse/profile=true`):

1. **`gcsfusecsi-training`**: Optimized for high read throughput during model training on GPUs/TPUs.
2. **`gcsfusecsi-checkpointing`**: Optimized for high write throughput to minimize training pauses during checkpoint saves.
3. **`gcsfusecsi-serving`**: Optimized for data access with default Rapid Cache (Anywhere Cache) for model serving.

### Empirical Profile Benchmark Findings (`n4-standard-80`, 1,650 Shards, ArrayRecord 155.27 GB):
- **Custom Inline CSI Mount Baseline**: Avg latency `19.69 ms`, p99 latency `39.60 ms`, throughput `3,743 samples/s`.
- **GKE `gcsfusecsi-training` Profile**: Avg latency `19.57 ms`, p99 latency `33.83 ms` (**p99 tail jitter reduced by 14.6%**), throughput `3,792 samples/s`.
- **Key Benefit**: Automated Google-managed kernel VFS and connection pool tuning without manual `mountOptions` maintenance.

---

## 🛠️ Technical Execution Protocols

### 0. Environment & Infrastructure Inspection Protocol
Before deploying workloads, extract detailed GKE cluster, compute node, hardware specs, GCSFuse CSI driver version, and VPC network parameters using the committed cluster manager tool:
```bash
python3 tools/infrastructure/cluster_manager.py --format=table
```
This automatically inspects and reports:
- **Node Machine Type & CPU/Mem Spec**: Instance type, CPU capacity, Memory (GiB)
- **OS & Container Runtime**: OS Image, Kernel version, Container runtime version
- **VPC Network MTU Configuration**: Network interface MTU (`8896 Jumbo Frames` vs `1460 Standard`)
- **GCSFuse CSI Driver Version**: Driver enabled status and image version tag (`v1.22.21-gke.1`)
- **JobSet CRD**: Installation status of `jobsets.jobset.x-k8s.io`

---

### 1. Multi-Process Parquet to ArrayRecord Transcoder & Manifest Generator

To saturate high-vCPU nodes (e.g. `n4-standard-80` with 80 vCPUs), the converter uses `ProcessPoolExecutor` to bypass the Python GIL and automatically generates a `manifest.json` metadata index:

```bash
# Multi-Process Transcode (Parquet -> ArrayRecord + manifest.json)
python3 tools/datasets/converters/parquet_to_arrayrecord.py \
  --input-path="gs://${BUCKET_NAME}/parquet_dataset" \
  --output-path="gs://${BUCKET_NAME}/arrayrecord_dataset" \
  --text-column="text" \
  --sequence-length=2048 \
  --num-workers=64

# Standalone Manifest Generation for Existing Datasets
python3 tools/infrastructure/bucket_manager.py \
  --action=generate-manifest \
  --dataset-uri="gs://${BUCKET_NAME}/arrayrecord_dataset"
```
* **Performance Baseline**: Reaches **~303 MB/s (118,000+ records/sec)**, transcoding 420 GB in under 18 minutes.
* **Manifest Output**: Automatically generates `manifest.json` in the destination directory to eliminate metadata listing storms.

---

### 2. Deploy Benchmark Runs via Helm

Navigate to `workloads/maxtext-dataset-loader/helm_chart`.

#### Mode A: Raw Storage Range Read Benchmark (`loaderMode=storage_bench`)
Evaluates raw file I/O bandwidth and seek latency:
```bash
helm install maxtext-storage-bench . \
  --set gcsfuse.enabled=true \
  --set gcsfuse.datasetBucket="${BUCKET_NAME}" \
  --set-string gcsfuse.mountOptions="implicit-dirs\,file-cache:max-size-mb:-1\,file-cache:cache-file-for-range-read:true" \
  --set gcsfs.datasetPath="/gcs/dataset" \
  --set workload.accessMode="gcsfuse" \
  --set workload.loaderMode="storage_bench" \
  --set workload.datasetFormat="arrayrecord" \
  --set workload.shuffleMode="two_stage" \
  --set workload.batchSize=128 \
  --set workload.maxBatches=500 \
  --set nodeSelector."cloud\.google\.com/gke-nodepool"="n4-standard-80"
```

#### Mode B: In-Tree Standalone DataLoader Benchmark (`loaderMode=in_tree_loader`)
Evaluates end-to-end data pipeline including **Two-Stage Shuffle**, **Buffer Priming**, and **CPU Tokenizer profiling**:
```bash
helm install maxtext-intree-bench . \
  --set gcsfuse.enabled=true \
  --set gcsfuse.datasetBucket="${BUCKET_NAME}" \
  --set-string gcsfuse.mountOptions="implicit-dirs\,file-cache:max-size-mb:-1\,file-cache:cache-file-for-range-read:true" \
  --set gcsfs.datasetPath="/gcs/dataset" \
  --set workload.accessMode="gcsfuse" \
  --set workload.loaderMode="in_tree_loader" \
  --set workload.datasetFormat="arrayrecord" \
  --set workload.useManifest="true" \
  --set workload.shuffleMode="two_stage" \
  --set workload.shuffleBufferSize=20000 \
  --set workload.numThreads=8 \
  --set workload.numStreams=8 \
  --set workload.batchSize=128 \
  --set workload.maxBatches=500 \
  --set nodeSelector."cloud\.google\.com/gke-nodepool"="n4-standard-80"
```

#### Mode C: Official GKE StorageClass Profile Benchmark (`gcsfuse.profile="training"`)
Evaluates the automated Google-tuned GCSFuse Training profile (`gcsfusecsi-training`):
```bash
helm install maxtext-profile-bench . \
  --set gcsfuse.enabled=true \
  --set gcsfuse.datasetBucket="${BUCKET_NAME}" \
  --set gcsfuse.profile="training" \
  --set gcsfs.datasetPath="/gcs/dataset" \
  --set workload.accessMode="gcsfuse" \
  --set workload.loaderMode="in_tree_loader" \
  --set workload.datasetFormat="arrayrecord" \
  --set workload.useManifest="true" \
  --set workload.shuffleMode="two_stage" \
  --set workload.shuffleBufferSize=20000 \
  --set workload.numStreams=8 \
  --set workload.batchSize=128 \
  --set workload.maxBatches=500 \
  --set nodeSelector."cloud\.google\.com/gke-nodepool"="n4-standard-80"
```

---

## 🔍 Cold-Start & Metadata Listing Storm Troubleshooting

If a user reports that initial dataset loading at training cluster startup takes tens of minutes:

1. **Root Cause: Metadata Listing Storm (`storage.objects.list` Throttle)**:
   * 1,000+ Pods concurrently calling `os.walk()` or `glob.glob()` on GCS causes HTTP 429/503 API throttling.
   * **Fix**: Ensure a single `manifest.json` exists in the dataset root directory. `standalone_dataloader.py` will read the single manifest directly, bypassing GCS tree walks.
2. **Root Cause: CPU Tokenizer Buffer Priming**:
   * Initializing a 20,000+ sample shuffle buffer with raw Parquet forces hundreds of CPU cores into intensive tokenization before Step 1 can start.
   * **Fix**: Pre-tokenize data into ArrayRecord offline.
3. **Root Cause: Missing GCSFuse Metadata Cache or Profile**:
   * Use `--set gcsfuse.profile="training"` to bind `gcsfusecsi-training`, or ensure `implicit-dirs,file-cache:max-size-mb:-1` is specified.
