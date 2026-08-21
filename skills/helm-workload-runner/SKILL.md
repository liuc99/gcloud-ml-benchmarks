---
name: helm-workload-runner
description: Sub-skill for constructing dynamic Helm chart flags, deploying benchmark releases on GKE (MaxText, PyTorch, TensorStore, multi-format loader), monitoring Pod states asynchronously until 100% completion, and handling normal and emergency teardown.
---

# Helm Workload Runner Sub-Skill (`helm-workload-runner`)

This sub-skill manages the deployment lifecycle of Helm workload releases, monitoring Pod execution milestones in the background, and performing both normal and emergency teardown.

---

## 🛠️ Tasks & Capabilities

### 1. Dynamic Helm Chart Command Construction

#### A. PyTorch Multi-Format Dataset Loader (`workloads/multi-format-dataset-loader/helm_chart`)

##### Managed Lustre Target:
```bash
helm install pytorch-loader-lustre . -f values_base.yaml \
  --set nodeSelector."cloud\.google\.com/gke-nodepool"=${NODEPOOL} \
  --set workload.nodes=${NODES:-1} \
  --set workload.ranksPerNode=${RANKS:-1} \
  --set workload.format="parquet" \
  --set workload.reader="hf_datasets" \
  --set workload.batchSize=64 \
  --set workload.numWorkers=4 \
  --set workload.prefetchFactor=2 \
  --set workload.maxBatches=100 \
  --set workload.epochs=1 \
  --set lustre.enabled=true \
  --set lustre.checkpointPvc="${LUSTRE_PVC}" \
  --set gcsfs.datasetPath="/lustre"
```

##### GCSFuse Target:
```bash
helm install pytorch-loader-gcsfuse . -f values_base.yaml \
  --set nodeSelector."cloud\.google\.com/gke-nodepool"=${NODEPOOL} \
  --set workload.nodes=${NODES:-1} \
  --set workload.ranksPerNode=${RANKS:-1} \
  --set workload.format="parquet" \
  --set workload.reader="hf_datasets" \
  --set workload.batchSize=64 \
  --set workload.numWorkers=4 \
  --set workload.prefetchFactor=2 \
  --set workload.maxBatches=100 \
  --set workload.epochs=1 \
  --set gcsfuse.enabled=true \
  --set gcsfuse.datasetBucket="${BUCKET_NAME}" \
  --set-string gcsfuse.mountOptions="implicit-dirs\,file-cache:max-size-mb:-1\,file-cache:cache-file-for-range-read:true" \
  --set gcsfs.datasetPath="/gcs/dataset"
```

##### Direct GCS `gcsfs` Target:
```bash
helm install pytorch-loader-gcsfs . -f values_base.yaml \
  --set nodeSelector."cloud\.google\.com/gke-nodepool"=${NODEPOOL} \
  --set workload.nodes=${NODES:-1} \
  --set workload.ranksPerNode=${RANKS:-1} \
  --set workload.format="parquet" \
  --set workload.reader="hf_datasets" \
  --set workload.batchSize=64 \
  --set workload.numWorkers=4 \
  --set workload.prefetchFactor=2 \
  --set workload.maxBatches=100 \
  --set workload.epochs=1 \
  --set gcsfs.datasetPath="gs://${BUCKET_NAME}"
```

---

#### B. PyTorch DDP Checkpointing (`workloads/hf-pytorch-lightning-cpu/helm_chart`)

##### Managed Lustre Target:
```bash
helm install pytorch-lustre-run . -f values_base.yaml \
  --set nodeSelector."cloud\.google\.com/gke-nodepool"=${NODEPOOL} \
  --set workload.nodes=${NODES} \
  --set workload.ranksPerNode=${RANKS} \
  --set workload.steps=${STEPS} \
  --set workload.ckptWriterInterval=${CKPT_INTERVAL} \
  --set workload.simulatedStepComputeSeconds=0.01 \
  --set lustre.enabled=true \
  --set lustre.checkpointPvc="${LUSTRE_PVC}" \
  --set lustre.mountPathCheckpoint="/lustre" \
  --set gcsfs.datasetPath="${DATASET_PATH}" \
  --set gcsfs.ckptWritePath="/lustre/checkpoints"
```

##### GCSFuse Target:
```bash
helm install pytorch-gcsfuse-run . -f values_base.yaml \
  --set nodeSelector."cloud\.google\.com/gke-nodepool"=${NODEPOOL} \
  --set workload.nodes=${NODES} \
  --set workload.ranksPerNode=${RANKS} \
  --set workload.steps=${STEPS} \
  --set workload.ckptWriterInterval=${CKPT_INTERVAL} \
  --set workload.simulatedStepComputeSeconds=0.01 \
  --set gcsfuse.enabled=true \
  --set gcsfuse.checkpointBucket="${BUCKET_NAME}" \
  --set gcsfuse.datasetBucket="${BUCKET_NAME}" \
  --set-string gcsfuse.mountOptions="implicit-dirs\,write:enable-streaming-writes:true\,write:global-max-blocks:-1" \
  --set gcsfs.datasetPath="/gcs/dataset" \
  --set gcsfs.ckptWritePath="/gcs/checkpoints/checkpoints_gcsfuse"
```

##### Direct GCS `gcsfs` Target:
```bash
helm install pytorch-gcsfs-run . -f values_base.yaml \
  --set nodeSelector."cloud\.google\.com/gke-nodepool"=${NODEPOOL} \
  --set workload.nodes=${NODES} \
  --set workload.ranksPerNode=${RANKS} \
  --set workload.steps=${STEPS} \
  --set workload.ckptWriterInterval=${CKPT_INTERVAL} \
  --set workload.simulatedStepComputeSeconds=0.01 \
  --set gcsfs.datasetPath="gs://${BUCKET_NAME}/dataset" \
  --set gcsfs.ckptWritePath="gs://${BUCKET_NAME}/checkpoints_gcsfs"
```

---

### 2. Asynchronous Execution Monitoring
- Monitor Pod status: `kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${RELEASE_NAME}`
- Use `schedule` background timers or async log streaming to monitor milestones:
  - Milestone 1: Dependency setup finished
  - Milestone 2: Dataset loaded into memory
  - Milestone 3: DDP training & Checkpoint write in progress
  - Milestone 4: Training completed
- Provide concise, professional status updates to the user as milestones are reached.

---

### 🚨 3. Emergency Abort & Teardown Protocol (Mid-Test Cancellation)

If the user sends a cancellation request ("stop", "cancel", "abort", "叫停", "停止") at any point during benchmark execution:

1. **Immediately Interrupt Matrix Loop**: Stop launching any subsequent test iterations or storage backend runs.
2. **Discover Active Helm Releases**:
   ```bash
   helm list -q
   ```
3. **Uninstall Active Release**:
   ```bash
   helm uninstall ${RELEASE_NAME} --ignore-not-found
   ```
4. **Confirm Safe Abort**:
   Inform the user:
   > *"Benchmark execution safely aborted. Active Helm release `${RELEASE_NAME}` has been uninstalled and cluster compute resources have been freed."*

---

### 4. Normal Release Teardown
Once metrics parsing is complete for a run, execute:
```bash
helm uninstall ${RELEASE_NAME}
```
Confirm to the orchestrator and user that the benchmark release has been uninstalled.
