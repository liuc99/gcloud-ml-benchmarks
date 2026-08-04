# ML Dataset Loading Benchmark & Demo Step-by-Step Guide

This guide walks you through deploying, benchmarking, and optimizing ML dataset ingestion performance across **Google Cloud Storage (GCSFuse & `gcsfs`)** and **Google Cloud Managed Lustre** on GKE.

---

## 📋 Prerequisites & Environment Setup

Set your GCP project, region, cluster name, and bucket variables:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export ZONE="us-central1-a"
export CLUSTER_NAME="ml-dataset-bench-cluster"
export BUCKET_NAME="ml-dataset-bench-${PROJECT_ID}"
```

---

## 🛠️ Step 1: Create Storage Bucket & GKE Cluster

### 1. Create Cloud Storage Bucket
```bash
gcloud storage buckets create gs://${BUCKET_NAME} \
  --project=${PROJECT_ID} \
  --location=${REGION} \
  --uniform-bucket-level-access
```

### 2. Create GKE Cluster with GCSFuse & Managed Lustre CSI Drivers
```bash
gcloud container clusters create ${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --release-channel=regular \
  --machine-type=c4-standard-192 \
  --num-nodes=2 \
  --addons=GcsFuseCsiDriver,LustreCsiDriver \
  --workload-pool=${PROJECT_ID}.svc.id.goog
```

---

## 🔐 Step 2: Configure Workload Identity IAM & JobSet CRD

```bash
# 1. Get Cluster Credentials
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone=${ZONE} --project=${PROJECT_ID}

# 2. Grant Storage Object Admin to default service account
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
gcloud storage buckets add-iam-policy-binding gs://${BUCKET_NAME} \
  --member="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/default/sa/default" \
  --role="roles/storage.objectAdmin"

# 3. Install JobSet CRD (v0.12.0)
kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/v0.12.0/manifests.yaml
```

---

## 🎲 Step 3: Generate Synthetic Benchmark Dataset on GCS

You can run the synthetic dataset generator container on GKE (or locally) to generate a 10 GB benchmark Parquet dataset directly to Cloud Storage:

```bash
# Option A: Generate directly over gs:// using gcsfs
python3 workloads/dataset-loading/helm_chart/dataset_generator.py \
  --output-path="gs://${BUCKET_NAME}/bench_dataset_parquet" \
  --format=parquet \
  --total-size-mb=10240 \
  --num-files=50
```

---

## 🚀 Step 4: Deploy Dataset Loading Benchmarks via Helm

Navigate to the dataset-loading Helm chart directory:

```bash
cd workloads/dataset-loading/helm_chart
```

### Scenario 1: GCSFuse Streaming Reads (CSI Sidecar Mount)

Test streaming read throughput over GCSFuse sidecar mounts (`/gcs/dataset`):

```bash
helm install bench-dataset-gcsfuse . -f values_base.yaml \
  --set workload.nodes=2 \
  --set workload.format="parquet" \
  --set workload.reader="hf_datasets" \
  --set workload.batchSize=64 \
  --set workload.numWorkers=8 \
  --set workload.maxBatches=500 \
  --set gcsfuse.enabled=true \
  --set gcsfuse.datasetBucket="${BUCKET_NAME}" \
  --set-string gcsfuse.mountOptions="implicit-dirs\,file-cache:max-size-mb:-1\,file-cache:cache-file-for-range-read:true" \
  --set gcsfs.datasetPath="/gcs/dataset/bench_dataset_parquet"
```

### Scenario 2: Native Python `gcsfs` Direct Streaming

Test direct GCS REST API ingestion using `gcsfs` and PyTorch DataLoader:

```bash
helm install bench-dataset-gcsfs . -f values_base.yaml \
  --set workload.nodes=2 \
  --set workload.format="parquet" \
  --set workload.reader="hf_datasets" \
  --set workload.batchSize=64 \
  --set workload.numWorkers=8 \
  --set workload.maxBatches=500 \
  --set gcsfs.datasetPath="gs://${BUCKET_NAME}/bench_dataset_parquet"
```

### Scenario 3: Google Cloud Managed Lustre Mount

Test parallel file system read performance over Managed Lustre (`/lustre/dataset`):

```bash
helm install bench-dataset-lustre . -f values_base.yaml \
  --set workload.nodes=2 \
  --set workload.format="parquet" \
  --set workload.reader="hf_datasets" \
  --set workload.batchSize=64 \
  --set workload.numWorkers=8 \
  --set workload.maxBatches=500 \
  --set lustre.enabled=true \
  --set lustre.checkpointPvc="lustre-pvc" \
  --set gcsfs.datasetPath="/lustre/dataset/bench_dataset_parquet"
```

---

## 📈 Step 5: Monitor Execution & View Benchmark Metrics

Monitor pod execution status and view stdout benchmark metrics:

```bash
# Watch pod execution
kubectl get pods -w

# Stream live benchmark output
kubectl logs -f -l jobset.sigs.k8s.io/jobset-name=bench-dataset-gcsfuse -c workload

# Parse metrics automatically via metrics parser
kubectl logs job/bench-dataset-gcsfuse-workload-0 -c workload | python3 skills/benchmark-metrics-parser/scripts/parse_metrics.py
```

### Example Parsed Metrics Output:

```json
{
  "ttfb_ms": 142.50,
  "dataset_read_throughput_mbs": 1845.20,
  "samples_per_sec": 128450.00,
  "p50_latency_ms": 3.42,
  "p95_latency_ms": 6.80
}
```

---

## 🧹 Step 6: Cleanup & Teardown

```bash
# Uninstall Helm releases
helm uninstall bench-dataset-gcsfuse bench-dataset-gcsfs bench-dataset-lustre

# Delete bucket & GKE Cluster
gcloud storage rm --recursive gs://${BUCKET_NAME}
gcloud container clusters delete ${CLUSTER_NAME} --zone=${ZONE} --project=${PROJECT_ID} --quiet
```
