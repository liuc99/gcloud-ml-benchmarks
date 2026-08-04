# MaxText Parquet GCS Range Read Benchmark & Demo Guide

This guide details how to generate multi-column Parquet datasets, deploy the MaxText dataset reader harness on GKE, and evaluate **GCS Range Read efficiency** across **Native GCS Client (`gs://...`)** and **GCSFuse Sidecar Mount (`/gcs/...`)**.

---

## 🛠️ Step 1: Environment & Bucket Setup

Set your environment variables:

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

# 3. Configure Workload Identity
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone=${ZONE} --project=${PROJECT_ID}
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")

gcloud storage buckets add-iam-policy-binding gs://${BUCKET_NAME} \
  --member="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/default/sa/default" \
  --role="roles/storage.objectAdmin"

# 4. Install JobSet CRD (v0.12.0)
kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/v0.12.0/manifests.yaml
```

---

## 🎲 Step 2: Generate Multi-Column Synthetic Parquet Dataset

Generate a 10 GB Parquet dataset containing `input_ids`, `attention_mask`, `label`, and a 4 KB/row `extra_metadata_bytes` column:

```bash
python3 workloads/maxtext-parquet-loader/helm_chart/dataset_generator.py \
  --output-path="gs://${BUCKET_NAME}/maxtext_parquet_dataset" \
  --total-size-mb=10240 \
  --num-files=20 \
  --sequence-length=2048 \
  --metadata-bytes-per-row=4096
```

---

## 🚀 Step 3: Run MaxText Parquet Range Read Benchmarks via Helm

Navigate to the Helm chart directory:

```bash
cd workloads/maxtext-parquet-loader/helm_chart
```

### Access Mode A: Native GCS Client (`gs://...` Direct REST/gRPC Range Requests)

Deploy using PyArrow native GCS filesystem (`gs://`):

```bash
helm install maxtext-native-gcs . -f values_base.yaml \
  --set workload.nodes=2 \
  --set workload.accessMode="native_gcs" \
  --set workload.columns="input_ids,label" \
  --set workload.batchSize=64 \
  --set workload.maxBatches=200 \
  --set gcsfs.datasetPath="gs://${BUCKET_NAME}/maxtext_parquet_dataset"
```

### Access Mode B: GCSFuse Sidecar Mount (`/gcs/...` Range-Read Caching)

Deploy using GCSFuse CSI sidecar mount with optimized range-read caching enabled:

```bash
helm install maxtext-gcsfuse . -f values_base.yaml \
  --set workload.nodes=2 \
  --set workload.accessMode="gcsfuse" \
  --set workload.columns="input_ids,label" \
  --set workload.batchSize=64 \
  --set workload.maxBatches=200 \
  --set gcsfuse.enabled=true \
  --set gcsfuse.datasetBucket="${BUCKET_NAME}" \
  --set-string gcsfuse.mountOptions="implicit-dirs\,file-cache:max-size-mb:-1\,file-cache:cache-file-for-range-read:true" \
  --set gcsfs.datasetPath="/gcs/dataset/maxtext_parquet_dataset"
```

---

## 📈 Step 4: Monitor Execution & Compare Range Read Performance

Check benchmark logs and analyze metrics:

```bash
# View container stdout output
kubectl logs -f -l jobset.sigs.k8s.io/jobset-name=maxtext-native-gcs -c workload
```

### Sample Performance Benchmark Output:

```
==================================================================================
                    MAXTEXT PARQUET GCS RANGE READ SUMMARY                        
==================================================================================
Access Mode              : native_gcs
Dataset Path             : gs://maxtext-parquet-bench/maxtext_parquet_dataset
Target Projected Columns : input_ids,label
Total Batches Ingested   : 200 batches
Total Samples Processed  : 12800 samples
Time to First Batch TTFB : 118.40 ms (0.1184 s)
Footer Metadata Latency  : 45.20 ms (20 range requests)
Data GCS Range Requests  : 200 requests
GCS Bytes Downloaded     : 215.04 MB
Useful Feature Payload   : 209.72 MB
Range Read Efficiency    : 97.53%
Read Throughput          : 1420.50 MB/s (11.36 Gbps)
Ingestion Speed          : 84500.00 samples/sec
Range Read Latency (Avg) : 2.15 ms
Range Read Latency (p95) : 4.80 ms
==================================================================================
```

---

## 💡 Range Read Optimization Insights for MaxText

1. **Bandwidth Savings**: Column projection (`--columns=input_ids,label`) avoids reading `extra_metadata_bytes`, resulting in **~70% reduction in GCS bandwidth & RAM consumption**.
2. **GCSFuse Tuning**: Setting `file-cache:cache-file-for-range-read:true` allows GCSFuse to cache Parquet footer byte ranges, lowering TTFB and eliminating repeated metadata RPC calls.
