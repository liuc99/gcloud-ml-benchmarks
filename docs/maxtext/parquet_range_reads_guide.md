# MaxText Parquet & ArrayRecord Dataset Benchmark & Demo Guide

This guide details how to generate multi-column Parquet datasets, convert raw Parquet to pre-tokenized **ArrayRecord** format, deploy the MaxText dataset reader harness on GKE, and evaluate **Parquet vs ArrayRecord** performance across **None**, **Two-Stage**, and **Global** shuffle modes.

---

## 🛠️ Step 1: Environment Setup

Set environment variables:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export ZONE="us-central1-a"
export CLUSTER_NAME="maxtext-gcs-bench-cluster"
export BUCKET_NAME="maxtext-parquet-bench-${PROJECT_ID}"
```

Create GCS bucket and GKE cluster:

```bash
# 1. Create Bucket
gcloud storage buckets create gs://${BUCKET_NAME} --project=${PROJECT_ID} --location=${REGION}

# 2. Create GKE Cluster with GCSFuse CSI Driver
gcloud container clusters create ${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --machine-type=c4-standard-192 \
  --num-nodes=2 \
  --addons=GcsFuseCsiDriver \
  --workload-pool=${PROJECT_ID}.svc.id.goog

# 3. Configure Workload Identity & Credentials
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone=${ZONE} --project=${PROJECT_ID}
```

---

## 🎲 Step 2: Generate Synthetic Parquet Dataset

Generate a 10 GB Parquet dataset containing `input_ids`, `label`, and metadata:

```bash
python3 workloads/maxtext-parquet-loader/helm_chart/dataset_generator.py \
  --output-path="gs://${BUCKET_NAME}/maxtext_parquet_dataset" \
  --total-size-mb=10240 \
  --num-files=20 \
  --sequence-length=2048 \
  --metadata-bytes-per-row=4096
```

---

## 🔄 Step 3: Pre-convert Parquet Dataset to ArrayRecord (Optional)

Convert raw Parquet text shards to pre-tokenized `.array_record` format holding `int32` token arrays using the standalone converter:

```bash
python3 workloads/maxtext-parquet-loader/helm_chart/parquet_to_arrayrecord.py \
  --input-path="gs://${BUCKET_NAME}/maxtext_parquet_dataset" \
  --output-path="gs://${BUCKET_NAME}/arrayrecord_dataset" \
  --sequence-length=2048 \
  --max-files=20
```

---

## 🚀 Step 4: Deploy Benchmark Runs via Helm

Navigate to the Helm chart directory:

```bash
cd workloads/maxtext-parquet-loader/helm_chart
```

### Benchmark Run A: Parquet Range Reads (Native GCS)
```bash
helm install maxtext-parquet-run . \
  --set gcsfs.datasetPath="gs://${BUCKET_NAME}/maxtext_parquet_dataset" \
  --set workload.datasetFormat="parquet" \
  --set workload.shuffleMode="two_stage" \
  --set workload.batchSize=64 \
  --set workload.maxBatches=100
```

### Benchmark Run B: ArrayRecord Streaming (Zero-CPU Bottleneck)
```bash
helm install maxtext-arrayrecord-run . \
  --set gcsfs.datasetPath="gs://${BUCKET_NAME}/arrayrecord_dataset" \
  --set workload.datasetFormat="arrayrecord" \
  --set workload.convertToArrayRecord=false \
  --set workload.shuffleMode="two_stage" \
  --set workload.batchSize=64 \
  --set workload.maxBatches=100
```

### Benchmark Run C: ArrayRecord Global True Random Shuffle
```bash
helm install maxtext-arrayrecord-global . \
  --set gcsfs.datasetPath="gs://${BUCKET_NAME}/arrayrecord_dataset" \
  --set workload.datasetFormat="arrayrecord" \
  --set workload.convertToArrayRecord=false \
  --set workload.shuffleMode="global" \
  --set workload.batchSize=64 \
  --set workload.maxBatches=100
```

---

## 📈 Step 5: Benchmark Results & Metrics Analysis

Check benchmark logs:

```bash
kubectl logs -f pod/maxtext-arrayrecord-run-workload-0-0-xxxxx
```

### Sample ArrayRecord Benchmark Output:

```
==================================================================================
               MAXTEXT ARRAYRECORD DATASET READ SUMMARY                 
==================================================================================
Data Format              : ARRAYRECORD
Access Mode              : native_gcs
Shuffle Mode             : two_stage
Dataset Path             : gs://my-bucket/arrayrecord_dataset
Total Dataset Shards     : 3 files
Target Projected Fields  : ['int32_tokens']
Total Batches Ingested   : 100 batches
Total Samples Ingested   : 6400 samples
Payload Data Read Volume : 16.57 MB (0.0162 GB)
Time to First Batch TTFB : 4713.36 ms (4.7134 s)
Schema Discovery Latency : 0.00 ms
IO Read Throughput       : 2.33 MB/s (0.02 Gbps)
Sample Ingestion Speed   : 898.78 samples/sec
Batch Load Latency (Avg) : 0.38 ms
Batch Load Latency (p50) : 0.35 ms
Batch Load Latency (p95) : 0.60 ms
Batch Load Latency (p99) : 0.68 ms
==================================================================================
```
