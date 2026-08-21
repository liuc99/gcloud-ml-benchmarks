# Step-by-Step Benchmark Reproduction Guide

This guide provides complete, end-to-end instructions for deploying Google Kubernetes Engine (GKE) clusters, creating Zonal RAPID GCS buckets with Hierarchical Namespace (HNS), configuring Workload Identity, deploying TensorStore + GCSFuse benchmark workloads, and collecting performance metrics.

---

## 📋 Prerequisites & Tooling

Before starting, ensure you have the following CLI tools installed and authenticated:
- `gcloud` (Google Cloud SDK)
- `kubectl` (v1.28+)
- `helm` (v3.0+)
- Active GCP Project with Compute Engine, GKE, and Storage APIs enabled.

---

## 🛠️ Step 1: Define Environment Variables

Set environment variables tailored to your GCP project and cluster requirements:

```bash
export PROJECT_ID="<YOUR_PROJECT_ID>"
export REGION="us-central1"
export ZONE="us-central1-b"
export CLUSTER_NAME="tensorstore-gcsfuse-cluster"
export BUCKET_NAME="${PROJECT_ID}-tensorstore-rapid-manual"
export GKE_SA="<YOUR_GCP_SERVICE_ACCOUNT>" # e.g. gcsfs-ci-runner@${PROJECT_ID}.iam.gserviceaccount.com
```

---

## 🪣 Step 2: Create Zonal RAPID GCS Bucket with HNS

Create a high-throughput Zonal **RAPID** bucket in `us-central1-b` with **Hierarchical Namespace (HNS)** enabled to optimize directory metadata operations and eliminate cross-zone transfer latency:

```bash
gcloud storage buckets create gs://${BUCKET_NAME} \
  --project=${PROJECT_ID} \
  --location=${REGION} \
  --placement=${ZONE} \
  --default-storage-class=RAPID \
  --enable-hierarchical-namespace \
  --uniform-bucket-level-access
```

---

## ☸️ Step 3: Create GKE Cluster with GCSFuse CSI Driver & 8896 MTU

Provision a GKE node pool using `n4-standard-80` nodes (80 vCPUs, 320 GB RAM, 50 Gbps physical NIC) with the GCSFuse CSI Driver enabled.

### Option A: 32-Node Cluster with 8896 Jumbo Frames (Terabit Scale)
```bash
# 1. Create VPC network with 8896 MTU enabled
gcloud compute networks create tensorstore-vpc \
  --project=${PROJECT_ID} \
  --subnet-mode=custom \
  --mtu=8896

gcloud compute networks subnets create tensorstore-subnet \
  --project=${PROJECT_ID} \
  --network=tensorstore-vpc \
  --region=${REGION} \
  --range="10.0.0.0/20"

# 2. Create GKE Cluster
gcloud container clusters create ${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --network=tensorstore-vpc \
  --subnetwork=tensorstore-subnet \
  --machine-type=n4-standard-80 \
  --num-nodes=32 \
  --addons=GcsFuseCsiDriver \
  --workload-pool=${PROJECT_ID}.svc.id.goog \
  --service-account="${GKE_SA}" \
  --scopes="https://www.googleapis.com/auth/cloud-platform" \
  --network-performance-configs="total-egress-bandwidth-tier=TIER_1"
```

### Option B: Single-Node Benchmark Baseline (50 Gbps NIC Saturation)
```bash
gcloud container clusters create ${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --machine-type=n4-standard-80 \
  --num-nodes=1 \
  --addons=GcsFuseCsiDriver \
  --workload-pool=${PROJECT_ID}.svc.id.goog \
  --service-account="${GKE_SA}" \
  --scopes="https://www.googleapis.com/auth/cloud-platform"
```

---

## 🔐 Step 4: Configure Workload Identity & IAM

Get `kubectl` cluster credentials and bind Storage Object Admin permissions to the Kubernetes default service account:

```bash
# 1. Authenticate kubectl
gcloud container clusters get-credentials ${CLUSTER_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID}

# 2. Add IAM policy binding for Workload Identity
export PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")

gcloud storage buckets add-iam-policy-binding gs://${BUCKET_NAME} \
  --member="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/default/sa/default" \
  --role="roles/storage.objectAdmin"
```

---

## ⚙️ Step 5: Install JobSet Controller

Install the JobSet controller CRD (v0.12.0) required to orchestrate multi-node distributed workload pods:

```bash
kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/v0.12.0/manifests.yaml
```

---

## 🚀 Step 6: Deploy Benchmark Workload via Helm

Navigate to the workload Helm chart in this repository:

```bash
cd workloads/tensorstore-gcsfuse/helm_chart
```

### Deployment Configuration Matrix

Select the command corresponding to the benchmark dimension you wish to test:

#### 1. Peak Cluster Read Throughput (HTTP/1.1 Protocol)
```bash
helm install tensorstore-bench . -f values_base.yaml \
  --set nodeSelector."cloud\.google\.com/gke-nodepool"=default-pool \
  --set workload.image="python:3.12-slim" \
  --set workload.tensorstoreShape="16000\,8000\,250" \
  --set workload.tensorstoreChunks="2000\,1000\,25" \
  --set workload.numWorkers=4 \
  --set gcsfuse.enabled=true \
  --set gcsfuse.checkpointBucket="${BUCKET_NAME}" \
  --set gcsfuse.mountOptions="implicit-dirs\,client-protocol=http1\,write:enable-streaming-writes:true\,write:global-max-blocks:-1"
```

#### 2. Peak Cluster Write Throughput (gRPC Protocol)
```bash
helm install tensorstore-bench . -f values_base.yaml \
  --set nodeSelector."cloud\.google\.com/gke-nodepool"=default-pool \
  --set workload.image="python:3.12-slim" \
  --set workload.tensorstoreShape="16000\,8000\,250" \
  --set workload.tensorstoreChunks="2000\,1000\,25" \
  --set workload.numWorkers=4 \
  --set gcsfuse.enabled=true \
  --set gcsfuse.checkpointBucket="${BUCKET_NAME}" \
  --set gcsfuse.mountOptions="implicit-dirs\,client-protocol=grpc\,write:enable-streaming-writes:true\,write:global-max-blocks:-1"
```

#### 3. Single-Node Optimal 8-Worker Baseline (200 MB Chunks)
```bash
helm install tensorstore-bench . -f values_base.yaml \
  --set nodeSelector."cloud\.google\.com/gke-nodepool"=default-pool \
  --set workload.image="python:3.12-slim" \
  --set workload.tensorstoreShape="16000\,8000\,250" \
  --set workload.tensorstoreChunks="2000\,1000\,25" \
  --set workload.numWorkers=8 \
  --set gcsfuse.enabled=true \
  --set gcsfuse.checkpointBucket="${BUCKET_NAME}" \
  --set gcsfuse.mountOptions="implicit-dirs\,client-protocol=grpc\,write:enable-streaming-writes:true\,write:global-max-blocks:-1"
```

---

## 📈 Step 7: Monitor Execution & Parse Log Results

Monitor job execution and view container log outputs:

```bash
# Watch pod execution status
kubectl get pods -w

# Stream live container benchmark logs
kubectl logs -f -l jobset.sigs.k8s.io/jobset-name=tensorstore-bench -c workload

# View completed benchmark results summary
kubectl logs job/tensorstore-bench-workload-0 -c workload
```

Expected log output format:
```text
[BENCHMARK] Aggregate Read finished in 22.012 sec | Size: 3906275.20 MB (3.81 TB) | Throughput: 172909.38 MB/s
[BENCHMARK] Aggregate Write finished in 32.610 sec | Size: 3906275.20 MB (3.81 TB) | Throughput: 114848.54 MB/s
```

---

## 🧹 Step 8: Teardown & Resource Cleanup

Clean up Helm workloads, cloud storage buckets, and GKE cluster resources after execution:

```bash
# Uninstall Helm workload release
helm uninstall tensorstore-bench

# Remove temporary GCS benchmark bucket
gcloud storage rm --recursive gs://${BUCKET_NAME}

# Teardown GKE Cluster
gcloud container clusters delete ${CLUSTER_NAME} --zone=${ZONE} --project=${PROJECT_ID} --quiet
```
