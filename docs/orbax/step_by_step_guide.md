# Orbax Checkpoint Resharding & Restore Step-by-Step Reproduction Guide

This guide provides complete instructions for setting up the cloud environment, running offline checkpoint resharding, and deploying the automated 100GB Orbax restore benchmark harness on Google Kubernetes Engine (GKE) and Google Cloud Storage (GCS).

---

## 📋 Prerequisites & Environment Setup

Set your environment variables:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export ZONE="us-central1-b"
export CLUSTER_NAME="orbax-bench-cluster"
export BUCKET_NAME="orbax-checkpoint-bench-${PROJECT_ID}"
```

---

## 🛠️ Step 1: Create GCS Bucket & GKE Cluster

### 1. Create Cloud Storage Bucket (Zonal RAPID recommended)
```bash
gcloud storage buckets create gs://${BUCKET_NAME} \
  --project=${PROJECT_ID} \
  --location=${REGION} \
  --uniform-bucket-level-access
```

### 2. Create GKE Cluster with GCSFuse CSI Driver & 8896 MTU
```bash
gcloud container clusters create ${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --machine-type=n4-standard-80 \
  --num-nodes=1 \
  --addons=GcsFuseCsiDriver \
  --workload-pool=${PROJECT_ID}.svc.id.goog
```

### 3. Configure Workload Identity & JobSet CRD
```bash
# Get cluster credentials
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone=${ZONE} --project=${PROJECT_ID}

# Grant Storage Object Admin to default service account
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
gcloud storage buckets add-iam-policy-binding gs://${BUCKET_NAME} \
  --member="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/default/sa/default" \
  --role="roles/storage.objectAdmin"

# Install JobSet CRD (v0.12.0)
kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/v0.12.0/manifests.yaml
```

---

## 🎲 Step 2: Offline Checkpoint Resharding via CLI

You can use the standalone CLI tool [`tools/checkpoints/orbax_reshard_rewriter.py`](../../tools/checkpoints/orbax_reshard_rewriter.py) to perform offline resharding on any GCS-mounted directory:

```bash
# Example: Reshard 5 source shards into 10 target worker shards
python3 tools/checkpoints/orbax_reshard_rewriter.py \
  --src-dir="/gcs/${BUCKET_NAME}/checkpoints_5shards/step_10000" \
  --dst-dir="/gcs/${BUCKET_NAME}/checkpoints_10shards/step_10000" \
  --strategy=dim_partitions \
  --dim-partitions="0:10" \
  --num-workers=16 \
  --verify
```

---

## 🚀 Step 3: Deploy Automated Benchmark via Helm

The benchmark Helm chart automates:
1. **Checkpoint Generation**: Creates 112 FSDP weight arrays (112.0 GB).
2. **CPU Offline Resharding**: Reshards 5-shard layout into 10-shard aligned layout.
3. **Multi-Worker Restore Evaluation**: Runs 3 timed restore runs comparing un-rewritten vs rewritten throughput.

```bash
# Deploy the benchmark release
helm install orbax-bench-100g workloads/orbax-checkpoint-benchmark/helm_chart \
  --set nodeSelector."cloud\.google\.com/gke-nodepool"=n4-standard-80 \
  --set gcsfuse.checkpointBucket="${BUCKET_NAME}" \
  --set workload.srcShards=5 \
  --set workload.dstWorkers=10 \
  --set workload.numLayers=16 \
  --set workload.hiddenDim=16384 \
  --set workload.numRuns=3 \
  --set workload.numWorkers=16
```

---

## 📈 Step 4: Monitor Execution & Capture Logs

```bash
# 1. Monitor Pod lifecycle until Completed
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=orbax-bench-100g -w

# 2. View stdout logs and performance summary table
kubectl logs -l jobset.sigs.k8s.io/jobset-name=orbax-bench-100g -c workload -f
```

---

## 🧹 Step 5: Teardown & Cleanup

```bash
# Uninstall Helm release
helm uninstall orbax-bench-100g

# Clean up bucket and cluster
gcloud storage rm --recursive gs://${BUCKET_NAME}
gcloud container clusters delete ${CLUSTER_NAME} --zone=${ZONE} --project=${PROJECT_ID} --quiet
```

---

## 🔗 Related Documentation
- [Orbax Workload Overview & Architecture](README.md)
- [100GB Checkpoint Resharding & Restore Benchmark Report](results/100gb_restore_acceleration.md)
- [Workload Helm Chart Reference](../../workloads/orbax-checkpoint-benchmark/README.md)
