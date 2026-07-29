# Step-by-Step Reproduction Guide: PyTorch DDP + Storage Benchmarks

This guide provides end-to-end instructions for deploying PyTorch Distributed Data Parallel (DDP) model training and checkpointing benchmarks against **Google Cloud Managed Lustre**, **GCSFuse**, and **gcsfs**.

---

## 📋 Workload Overview

The PyTorch benchmark (`hf-pytorch-lightning-cpu`) simulates training a **Llama 3.1 8B** model across multiple GKE compute nodes. It evaluates:
- **Checkpoint Save Latency**: Time to serialize and persist model weights and AdamW optimizer state (~16 GB per rank).
- **Restore / Resume I/O Speed**: Time to restore checkpoint state during fault recovery.
- **Dataset Read Throughput**: Dataloader workers reading training data files across Lustre PVCs, GCSFuse mounts, or `gcsfs`.

---

## 🛠️ Step 1: Define Environment Variables

```bash
export PROJECT_ID="<YOUR_PROJECT_ID>"
export REGION="us-central1"
export ZONE="us-central1-b"
export CLUSTER_NAME="pytorch-storage-cluster"
export BUCKET_NAME="${PROJECT_ID}-pytorch-checkpoint-bucket"
export GKE_SA="<YOUR_GCP_SERVICE_ACCOUNT>"
```

---

## 💾 Step 2: Running PyTorch over Google Cloud Managed Lustre

Google Cloud Managed Lustre provides parallel POSIX storage for high-throughput PyTorch training.

### 1. Provision GKE Cluster with Lustre CSI Driver
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

### 2. Submit CloudBuild Managed Lustre Benchmark Job
```bash
gcloud builds submit \
  --config=cloudbuild/macrobenchmarks-gcsfuse-cloudbuild.yaml \
  --substitutions=\
_USE_LUSTRE="true",\
_LUSTRE_INSTANCE="<YOUR_LUSTRE_INSTANCE_ID>",\
_NODES="2",\
_RANKS_PER_NODE="4",\
_STEPS="100",\
_CHECKPOINT_INTERVAL="25"
```

The build script automatically queries the Managed Lustre instance IP and filesystem via `gcloud alpha lustre instances describe`, then provisions the Kubernetes PersistentVolume (PV) and PersistentVolumeClaim (PVC) using `driver: lustre.csi.storage.gke.io`.

---

## 🪣 Step 3: Running PyTorch over GCSFuse

To run PyTorch model training and streaming writes over GCSFuse sidecar mounts:

```bash
gcloud builds submit \
  --config=cloudbuild/macrobenchmarks-gcsfuse-cloudbuild.yaml \
  --substitutions=\
_USE_GCSFUSE="true",\
_GCSFUSE_ENABLE_STREAM_WRITE="true",\
_GCSFUSE_MOUNT_OPTIONS="implicit-dirs\,write:enable-streaming-writes:true\,write:global-max-blocks:-1",\
_NODES="2",\
_RANKS_PER_NODE="4",\
_STEPS="100",\
_CHECKPOINT_INTERVAL="25"
```

---

## 🐍 Step 4: Running PyTorch over gcsfs (Native Python Client)

To run PyTorch model checkpointing using native Python `gcsfs`:

```bash
gcloud builds submit \
  --config=cloudbuild/macrobenchmarks-cloudbuild.yaml \
  --substitutions=\
_WORKLOAD="hf-pytorch-lightning-cpu",\
_NODES="2",\
_RANKS_PER_NODE="4",\
_STEPS="100",\
_CHECKPOINT_INTERVAL="25"
```

---

## 📊 Step 5: Viewing Benchmark Results & Log Outputs

View the container training logs and checkpoint metrics:

```bash
# Get credentials
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone=${ZONE} --project=${PROJECT_ID}

# View workload pod logs
kubectl logs -l jobset.sigs.k8s.io/jobset-name=<RUN_ID> -c workload --tail=1000
```
