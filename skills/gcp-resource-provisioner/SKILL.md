---
name: gcp-resource-provisioner
description: Sub-skill for discovering existing GCP/GKE resources or provisioning GKE clusters, Managed Lustre instances, GCS buckets, and Workload Identity IAM bindings with strict resource lifetime guardrails.
---

# GCP Resource Provisioner Sub-Skill (`gcp-resource-provisioner`)

This sub-skill handles environment discovery, cluster credentials authentication, and cloud resource lifecycle management.

---

## 🔒 STRICT RESOURCE SAFETY & TEARDOWN GUARDRAILS

### 1. Persistent User Resources (NEVER DELETE)
- **Pre-existing GKE Clusters** (e.g. `chongliu-gke-persistent`)
- **Pre-existing Managed Lustre Instances / PVCs** (e.g. `lustre-checkpoint-pvc`)
- **Pre-existing GCS Buckets** (e.g. `chongliu-macrobench-dataset-f038a966`)

**Rule**: If a resource existed prior to the benchmark run or was explicitly supplied by the user as a persistent asset, **NEVER delete, modify, or tear down this resource**.

### 2. Ad-hoc Ephemeral Benchmark Resources (DESTROY IMMEDIATELY AFTER RUN)
- **Helm Workload Releases**: JobSet Pods created specifically for the benchmark run (`helm uninstall`).
- **Temporary Ephemeral GKE Test Clusters**: Created when user has no existing cluster (`gcloud container clusters create-auto temp-benchmark-cluster`). **MUST be deleted via `gcloud container clusters delete` immediately after benchmark completion.**
- **Temporary Ephemeral GCS Buckets / PVCs**: Created explicitly by the agent *during* the run as transient test artifacts.

---

## 🛠️ Tasks & Capabilities

### 1. Environment Auto-Discovery
- **Check GKE Credentials**: Run `kubectl config current-context` or `gcloud container clusters get-credentials`.
- **Check Lustre PVC**: Run `kubectl get pvc` to verify if a Managed Lustre PVC (e.g. `lustre-checkpoint-pvc`) exists in the target namespace.
- **Check GCS Bucket**: Run `gcloud storage buckets list` to verify if the requested GCS bucket exists.

### 2. Optional Resource Provisioning (If User Requested New Resources)
- **Create GCS Bucket**:
  ```bash
  gcloud storage buckets create gs://${BUCKET_NAME} --project=${PROJECT_ID} --location=${REGION} --uniform-bucket-level-access
  ```
- **Create Managed Lustre Instance**:
  ```bash
  gcloud alpha lustre instances create ${LUSTRE_INSTANCE} --project=${PROJECT_ID} --location=${ZONE} --capacity-gib=12000 --network=default
  ```
- **Create K8s PV/PVC for Lustre**:
  ```bash
  export LUSTRE_IP=$(gcloud alpha lustre instances describe ${LUSTRE_INSTANCE} --location=${ZONE} --format="value(networkConfig.ipAddress)")
  export LUSTRE_FS=$(gcloud alpha lustre instances describe ${LUSTRE_INSTANCE} --location=${ZONE} --format="value(filesystemName)")
  cat <<EOF | kubectl apply -f -
  apiVersion: v1
  kind: PersistentVolume
  metadata:
    name: ${LUSTRE_PVC}-pv
  spec:
    accessModes: [ReadWriteMany]
    capacity: {storage: 10Ti}
    csi:
      driver: lustre.csi.storage.gke.io
      volumeHandle: "${PROJECT_ID}/${ZONE}/${LUSTRE_INSTANCE}"
      volumeAttributes: {ip: "${LUSTRE_IP}", filesystem: "${LUSTRE_FS}"}
    storageClassName: ""
    persistentVolumeReclaimPolicy: Retain
  ---
  apiVersion: v1
  kind: PersistentVolumeClaim
  metadata: {name: ${LUSTRE_PVC}}
  spec:
    accessModes: [ReadWriteMany]
    storageClassName: ""
    resources: {requests: {storage: 10Ti}}
    volumeName: ${LUSTRE_PVC}-pv
  EOF
  ```
- **Configure Workload Identity IAM**:
  ```bash
  gcloud storage buckets add-iam-policy-binding gs://${BUCKET_NAME} \
    --member="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/default/sa/default" \
    --role="roles/storage.objectAdmin"
  ```

---

## 💻 3. Recommended GKE Machine Types & Node Pool Selection Matrix

| Benchmark Workload Scenario | Recommended Machine Type / Spec | Network Egress Bandwidth | Key Provisioning Flags & Rationale |
| :--- | :--- | :--- | :--- |
| **Pure I/O & Dataset Ingestion** *(Parquet/ArrayRecord Loading)* | **`n4-standard-80`** or **`c3-standard-88`** | **High Egress (~50 - 100 Gbps)** | `--machine-type=n4-standard-80 --network-performance-configs=total-egress-bandwidth-tier=TIER_1`<br>*High CPU & network throughput without expensive GPU/TPU quota.* |
| **MaxText JAX LLM Training** *(TPU Acceleration)* | **`ct6e-standard-4t`** *(v6e/Trillium)* or **`ct5p-hightpu-4t`** *(v5p)* | **Optical ICI Mesh + High GCS Bandwidth** | `--node-locations=${ZONE} --machine-type=ct6e-standard-4t`<br>*Ultra-high throughput ICI mesh for distributed JAX training.* |
| **PyTorch DDP LLM Training** *(GPU Acceleration)* | **`a3-highgpu-8g`** *(H100)* or **`a2-ultragpu-8g`** *(A100)* | **3.2 Tbps GPUDirect RDMA** | `--accelerator=type=nvidia-h100-80gb,count=8`<br>*Massive GPU memory and inter-GPU bandwidth.* |
| **Cost-Effective Entry Baseline** *(Lightweight Demo)* | **`n2-standard-16`** or **`e2-standard-16`** | Standard Bandwidth (~16 Gbps) | `--machine-type=n2-standard-16`<br>*Low-cost baseline for single-node debugging.* |
