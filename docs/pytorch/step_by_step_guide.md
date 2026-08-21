# Step-by-Step Reproduction Guide: PyTorch DDP + Storage Benchmarks

This guide provides complete, end-to-end instructions for deploying Google Kubernetes Engine (GKE) clusters, configuring Workload Identity, deploying PyTorch Distributed Data Parallel (DDP) model training & checkpointing benchmarks against **Google Cloud Managed Lustre**, **GCSFuse**, and **`gcsfs`**, and collecting performance metrics.

Workloads can be executed either interactively via our **AI Agent Skills** (`ml-benchmark-orchestrator`) or directly via **Helm**.

---

## 📋 Prerequisites & Tooling

Before starting, ensure you have the following CLI tools installed and authenticated:
- `gcloud` (Google Cloud SDK)
- `kubectl` (v1.28+)
- `helm` (v3.0+)
- Active GCP Project with Compute Engine, GKE, and Storage APIs enabled.

---

## 📋 Workload Overview

The PyTorch benchmark (`hf-pytorch-lightning-cpu`) simulates training a **Llama 3.1 8B** model across multiple GKE compute nodes. It evaluates:
- **Checkpoint Save Latency**: Time to serialize and persist model weights and AdamW optimizer state (~16 GB per rank).
- **Restore / Resume I/O Speed**: Time to restore checkpoint state during fault recovery.
- **Dataset Read Throughput**: Dataloader workers reading training data files across Lustre PVCs, GCSFuse mounts, or `gcsfs`.

---

## 🛠️ Step 1: Define Environment Variables

Set environment variables tailored to your GCP project and cluster requirements:

```bash
export PROJECT_ID="<YOUR_PROJECT_ID>"
export REGION="us-central1"
export ZONE="us-central1-b"
export CLUSTER_NAME="pytorch-storage-cluster"
export BUCKET_NAME="${PROJECT_ID}-pytorch-checkpoint-bucket"
export GKE_SA="<YOUR_GCP_SERVICE_ACCOUNT>" # e.g. gcsfs-ci-runner@${PROJECT_ID}.iam.gserviceaccount.com
export PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
export DATASET_PATH="gs://${BUCKET_NAME}/dataset" # Path to training Parquet dataset files

# Managed Lustre Variables (Required only if benchmark testing Google Cloud Managed Lustre)
export LUSTRE_INSTANCE="pytorch-lustre-instance"
export LUSTRE_PVC="lustre-checkpoint-pvc"
export LUSTRE_IP=$(gcloud alpha lustre instances describe ${LUSTRE_INSTANCE} --location=${ZONE} --format="value(networkConfig.ipAddress)" 2>/dev/null || echo "")
export LUSTRE_FS=$(gcloud alpha lustre instances describe ${LUSTRE_INSTANCE} --location=${ZONE} --format="value(filesystemName)" 2>/dev/null || echo "")
```

---

## 🪣 Step 2: Create GCS Checkpoint Bucket

Create a GCS bucket for model checkpoints and datasets:

```bash
gcloud storage buckets create gs://${BUCKET_NAME} \
  --project=${PROJECT_ID} \
  --location=${REGION} \
  --uniform-bucket-level-access
```

---

## 💾 Step 3: Provision Google Cloud Managed Lustre Instance (Optional)

If evaluating **Google Cloud Managed Lustre** and you do not have an existing instance, provision a new Managed Lustre parallel file system instance:

```bash
gcloud alpha lustre instances create ${LUSTRE_INSTANCE} \
  --project=${PROJECT_ID} \
  --location=${ZONE} \
  --capacity-gib=12000 \
  --network=default
```

*Note: Once created, update `LUSTRE_IP` and `LUSTRE_FS` in Step 1 using `gcloud alpha lustre instances describe`.*

---

## ☸️ Step 4: Create GKE Cluster with Lustre & GCSFuse CSI Drivers

Provision a GKE node pool using `c4-standard-192` nodes (192 vCPUs, 720 GB RAM) with both the Managed Lustre CSI Driver (`LustreCsiDriver`) and GCSFuse CSI Driver (`GcsFuseCsiDriver`) enabled:

```bash
gcloud container clusters create ${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --machine-type=c4-standard-192 \
  --num-nodes=2 \
  --addons=LustreCsiDriver,GcsFuseCsiDriver \
  --workload-pool=${PROJECT_ID}.svc.id.goog \
  --service-account="${GKE_SA}"
```

---

## 🔐 Step 5: Configure Workload Identity & IAM

Get `kubectl` cluster credentials and bind Storage Object Admin permissions to the Kubernetes default service account:

```bash
# 1. Authenticate kubectl
gcloud container clusters get-credentials ${CLUSTER_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID}

# 2. Add IAM policy binding for Workload Identity
gcloud storage buckets add-iam-policy-binding gs://${BUCKET_NAME} \
  --member="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/default/sa/default" \
  --role="roles/storage.objectAdmin"
```

---

## ⚙️ Step 6: Install JobSet Controller

Install the JobSet controller CRD (v0.12.0) required to orchestrate multi-node PyTorch DDP training jobs:

```bash
kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/v0.12.0/manifests.yaml
```

---

## 🚀 Step 7: Deploy Benchmark Workload

You can deploy PyTorch benchmark jobs across three different storage backends (**Google Cloud Managed Lustre**, **GCSFuse**, and **`gcsfs`**). You can run this either **interactively with your AI Agent** (recommended) or **directly via Helm**.

---

### Option A: Interactive AI Agent & Skill Execution (Recommended)

Simply prompt your AI Agent to execute the benchmark. The agent will invoke `ml-benchmark-orchestrator` to align on your parameters, confirm an execution plan table, pre-flight the environment, deploy the workload, and output a parsed performance comparison report:

```text
"Run a comparative PyTorch DDP benchmark on 2 nodes comparing Google Cloud Managed Lustre, GCSFuse streaming writes, and direct gcsfs checkpointing."
```

---

### Option B: Direct Helm Deployment

Navigate to the workload Helm chart directory in this repository:

```bash
cd workloads/hf-pytorch-lightning-cpu/helm_chart
```

#### 1. Google Cloud Managed Lustre

> 💡 **Managed Lustre PVC Prerequisite**:  
> Managed Lustre mounts require a pre-existing Kubernetes `PersistentVolumeClaim` (PVC) bound to your Managed Lustre instance via `driver: lustre.csi.storage.gke.io`.

*(Optional)* Create the PersistentVolume and PersistentVolumeClaim if not already created in your cluster:

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolume
metadata:
  name: ${LUSTRE_PVC}-pv
spec:
  accessModes:
  - ReadWriteMany
  capacity:
    storage: 10Ti
  csi:
    driver: lustre.csi.storage.gke.io
    volumeHandle: "${PROJECT_ID}/${ZONE}/${LUSTRE_INSTANCE}"
    volumeAttributes:
      ip: "${LUSTRE_IP}"
      filesystem: "${LUSTRE_FS}"
  storageClassName: ""
  persistentVolumeReclaimPolicy: Retain
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${LUSTRE_PVC}
spec:
  accessModes:
  - ReadWriteMany
  storageClassName: ""
  resources:
    requests:
      storage: 10Ti
  volumeName: ${LUSTRE_PVC}-pv
EOF
```

Deploy via Helm:

```bash
helm install pytorch-lustre . -f values_base.yaml \
  --set nodeSelector."cloud\.google\.com/gke-nodepool"=default-pool \
  --set workload.nodes=2 \
  --set workload.ranksPerNode=4 \
  --set workload.steps=100 \
  --set workload.ckptWriterInterval=25 \
  --set lustre.enabled=true \
  --set lustre.checkpointPvc="${LUSTRE_PVC}" \
  --set gcsfs.datasetPath="/lustre/dataset" \
  --set gcsfs.ckptWritePath="/lustre/checkpoints"
```

#### 2. GCSFuse Sidecar Mounts

Deploy GCSFuse sidecar streaming writes via Helm:

```bash
helm install pytorch-gcsfuse . -f values_base.yaml \
  --set nodeSelector."cloud\.google\.com/gke-nodepool"=default-pool \
  --set workload.nodes=2 \
  --set workload.ranksPerNode=4 \
  --set workload.steps=100 \
  --set workload.ckptWriterInterval=25 \
  --set gcsfuse.enabled=true \
  --set gcsfuse.checkpointBucket="${BUCKET_NAME}" \
  --set gcsfuse.datasetBucket="${BUCKET_NAME}" \
  --set-string gcsfuse.mountOptions="implicit-dirs\,write:enable-streaming-writes:true\,write:global-max-blocks:-1" \
  --set gcsfs.datasetPath="/gcs/dataset" \
  --set gcsfs.ckptWritePath="/gcs/checkpoints/checkpoints_gcsfuse"
```

#### 3. Native Python `gcsfs` Client

Deploy native Python `gcsfs` checkpointing via Helm:

```bash
helm install pytorch-gcsfs . -f values_base.yaml \
  --set nodeSelector."cloud\.google\.com/gke-nodepool"=default-pool \
  --set workload.nodes=2 \
  --set workload.ranksPerNode=4 \
  --set workload.steps=100 \
  --set workload.ckptWriterInterval=25 \
  --set gcsfs.datasetPath="${DATASET_PATH}" \
  --set gcsfs.ckptWritePath="gs://${BUCKET_NAME}/checkpoints_gcsfs"
```

---

## 📈 Step 8: Monitor Execution & View Log Outputs

Monitor job execution status and view PyTorch container benchmark logs:

```bash
# Watch pod execution status
kubectl get pods -w

# Stream live container benchmark logs (replace <RELEASE_NAME> with e.g. pytorch-lustre, pytorch-gcsfuse, or pytorch-gcsfs)
kubectl logs -f -l jobset.sigs.k8s.io/jobset-name=<RELEASE_NAME> -c workload

# View completed benchmark results summary
kubectl logs job/<RELEASE_NAME>-workload-0 -c workload
```

---

## 🧹 Step 9: Teardown & Resource Cleanup

Clean up Helm workload releases, Kubernetes PV/PVCs, cloud storage buckets, and GKE cluster resources:

```bash
# Uninstall Helm workload release
helm uninstall pytorch-lustre   # or pytorch-gcsfuse / pytorch-gcsfs

# Delete Lustre PVC and PV (if created manually)
kubectl delete pvc ${LUSTRE_PVC} --ignore-not-found
kubectl delete pv ${LUSTRE_PVC}-pv --ignore-not-found

# Delete Managed Lustre Instance (if created manually)
gcloud alpha lustre instances delete ${LUSTRE_INSTANCE} --location=${ZONE} --project=${PROJECT_ID} --quiet

# Remove temporary GCS benchmark bucket
gcloud storage rm --recursive gs://${BUCKET_NAME}

# Teardown GKE Cluster
gcloud container clusters delete ${CLUSTER_NAME} --zone=${ZONE} --project=${PROJECT_ID} --quiet
```
