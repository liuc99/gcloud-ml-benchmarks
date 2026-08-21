---
name: gcp-resource-provisioner
description: Sub-skill for discovering existing GCP/GKE resources, running cluster pre-flight diagnostics (tools/infrastructure/cluster_manager.py), managing GCS bucket lifecycles (tools/infrastructure/bucket_manager.py), and enforcing strict persistent asset protection.
---

# GCP Resource Provisioner Sub-Skill (`gcp-resource-provisioner`)

This sub-skill handles environment discovery, cluster credentials authentication, and cloud resource lifecycle management.

---

## 🔒 STRICT RESOURCE SAFETY & TEARDOWN GUARDRAILS

### 1. Persistent User Resources & Datasets (NEVER DELETE)
- **Pre-existing GKE Clusters** (e.g. `gke-persistent-cluster`)
- **Pre-existing Managed Lustre Instances / PVCs** (e.g. `lustre-checkpoint-pvc`)
- **Pre-existing GCS Buckets** (e.g. `gs://<user-dataset-bucket>`, pre-existing dataset buckets)
- **Pre-existing Datasets / Files**: Any dataset or files provided by the user or existing prior to the benchmark run.

> [!CAUTION]
> **ABSOLUTE IMMUTABILITY RULE**: User-supplied or pre-existing GCS buckets AND user-supplied or pre-existing datasets MUST NEVER BE DELETED under any circumstances!
> - ⛔ **PERSISTENT BUCKETS & DATASETS ARE 100% IMMUTABLE**: Deleting user-provided buckets or pre-existing dataset files is STRICTLY PROHIBITED.
> - ✅ **ONLY AGENT-CREATED EPHEMERAL ASSETS**: Only buckets created by the agent *during* the current run and ephemeral Helm workload releases (`helm uninstall <RELEASE>`) may be cleaned up.

### 2. Ad-hoc Ephemeral Benchmark Resources (CLEAN UP AFTER RUN)
- **Helm Workload Releases**: Ephemeral JobSet Pods created specifically for the benchmark run (`helm uninstall <RELEASE>`).
- **Temporary Ephemeral GKE Test Clusters**: Created *only* when user has no existing cluster. **MUST be deleted via `gcloud container clusters delete` immediately after run.**
- **Temporary Ephemeral GCS Buckets**: Created explicitly by the agent *during* the current run as transient test artifacts. Clean up via formal tool:
  ```bash
  python3 tools/infrastructure/bucket_manager.py --action=delete --project-id="${PROJECT_ID}" --bucket-name="${BUCKET_NAME}"
  ```
- **Temporary Test Prefix Cleanup**: Clean up ephemeral prefix paths inside buckets via formal tool:
  ```bash
  python3 tools/infrastructure/bucket_manager.py --action=cleanup-prefix --project-id="${PROJECT_ID}" --bucket-name="${BUCKET_NAME}" --prefix="${PREFIX_PATH}"
  ```

### 3. Strict Execution Policy: Formal Repository CLI Tools Only
- ⛔ **NO INLINE ON-THE-FLY PYTHON SCRIPTS**: NEVER run `python3 -c "..."` or write uncommitted inline code on the fly to provision, inspect, or clean up resources.
- ✅ **USE FORMAL REPO TOOLS ONLY**: All resource creation, inspection, dataset generation, and resource cleanup MUST invoke formal Python CLI tools (`tools/infrastructure/bucket_manager.py`, `tools/infrastructure/cluster_manager.py`, `tools/datasets/generator.py`).

### 4. Strict Prohibition Against Autonomous Bucket Scanning
- ⛔ **NO PROACTIVE BUCKET DISCOVERY/SCANNING**: The agent MUST NOT run `bucket_manager.py --action=list`, `resolve-existing`, or `gcloud storage ls` to scan or list buckets in the project without explicit user instruction.
- ✅ **USER-PROVIDED OR USER-DIRECTED ONLY**: Target GCS buckets MUST be explicitly provided by the user or created upon explicit user direction.

---

## 🛠️ Tasks & Capabilities

### 1. Environment Auto-Discovery
- **Run Environment & Dependency Pre-flight Check**:
  ```bash
  python3 tools/infrastructure/env_checker.py --format=table
  ```
  *(If missing CLI tools or Python libraries are reported, DO NOT fix them autonomously. Present a structured Remediation Plan table with exact installation commands for user review and approval before executing.)*
- **Run Pre-flight Cluster Diagnostics**:
  ```bash
  python3 tools/infrastructure/cluster_manager.py \
    --cluster-name "${CLUSTER_NAME}" \
    --zone "${ZONE}" \
    --project-id "${PROJECT_ID}"
  ```
- **Check Lustre PVC**: Run `kubectl get pvc` to verify if a Managed Lustre PVC (e.g. `lustre-checkpoint-pvc`) exists in the target namespace.

### 2. Optional Resource Provisioning (If User Requested New Resources)
- **Create / Ensure GCS Bucket**:
  Use the deterministic Python helper script (handles ADC authentication, Regional/Zonal/RAPID/HNS configuration, and auto-discovery):
  ```bash
  python3 tools/infrastructure/bucket_manager.py \
    --action ensure \
    --project-id "${PROJECT_ID}" \
    --bucket-type "${BUCKET_TYPE:-regional}" \
    --location "${REGION:-us-central1}" \
    --zone "${ZONE:-us-central1-b}" \
    --bucket-name "${BUCKET_NAME}"
  ```
  *(Or fallback to `gcloud storage buckets create gs://${BUCKET_NAME} --project=${PROJECT_ID} --location=${REGION} --uniform-bucket-level-access`)*
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
