# Agent Behavior Rules & Guardrails (`AGENTS.md`)

## 🛑 MANDATORY RULE 1: SKILL-FIRST PROTOCOL (NEVER FREESTYLE)
Whenever the user proposes or requests running any benchmark, performance test, dataset loading comparison, GCS storage evaluation, or workload execution:
1. **ALWAYS SEARCH & VIEW SKILLS FIRST**: The Agent MUST view and follow the relevant skills in `skills/` (e.g. `skills/ml-benchmark-orchestrator/SKILL.md` and `skills/maxtext-dataset-benchmark/SKILL.md`) using `view_file` BEFORE executing any commands or deployments.
2. **NO AD-HOC INLINE SCRIPTS**: Strict prohibition against executing inline Python (`python3 -c "..."`) or temporary ad-hoc scripts. All cluster diagnostics, bucket management, and dataset inspection MUST use formal committed CLI tools in `tools/` (`tools/infrastructure/cluster_manager.py`, `tools/infrastructure/bucket_manager.py`).

---

## 📋 MANDATORY RULE 2: INTERACTIVE PLAN CONFIRMATION BEFORE EXECUTION (INITIAL & SUPPLEMENTAL)
1. **INTERACTIVE ALIGNMENT FIRST**: Always conduct interactive questionnaire alignment to confirm target dataset path, format, shuffle strategies, and access modes.
2. **STRUCTURED PLAN REVIEW TABLE**: Present a structured Markdown table detailing:
   - Workload & Model under test
   - Target GKE cluster and node specs
   - GCSFuse CSI Driver version & VPC MTU
   - Storage backends and configuration flags under test
   - Input dataset overview (total size, shard count, schema)
   - Cloud resource consumption estimate
3. **APPLIES TO ALL INCREMENTAL / SUPPLEMENTAL RUNS**: Whenever the user asks to supplement, add, or vary a test case (e.g. "补充DirectGCS的数据", "再测一下不用manifest", "换一种shuffle策略"), NEVER treat it as an immediate run ticket. ALWAYS present an updated / supplemental Execution Plan Review table first.
4. **STRICT NO-COMMAND FIRST TURN & EXPLICIT USER APPROVAL**: The Agent is strictly forbidden from executing `helm install`, `kubectl apply`, or benchmark workload commands in the same turn that a benchmark or supplemental test is introduced. The Agent MUST pause and wait for explicit user confirmation (e.g., "Proceed" / "确认") before invoking sub-skills, creating resources, or running Helm/kubectl deployments.

---

## 🔒 MANDATORY RULE 3: STRICT PERSISTENT RESOURCE & DATASET PROTECTION
1. **IMMUTABLE PERSISTENT ASSETS**: Never delete, overwrite, or mutate pre-existing GCS buckets, persistent GKE clusters, or existing datasets.
2. **DATASET PARITY**: In comparative evaluations, all storage backends/configurations MUST be benchmarked against identical dataset dimensions and schemas (no silent substitution).
3. **MANDATORY COMPLETE POD LIFECYCLE MONITORING**: Never teardown Helm releases while pods are still running. Only teardown ephemeral Helm releases after workload pods reach `Completed` status and logs are captured.

---

## 🚫 MANDATORY RULE 4: STRICT PROHIBITION ON AUTONOMOUS BUCKET SCANNING (NEVER PROACTIVELY SCAN / LIST BUCKETS)
1. **USER-PROVIDED OR USER-DIRECTED BUCKETS ONLY**:
   - Target GCS Buckets MUST be explicitly supplied by the user (e.g. `gs://<user-bucket>/...`), OR
   - Created strictly upon explicit user command/instruction (e.g. user directs the agent to create a new bucket with specific name/parameters).
2. **NEVER SCAN OR LIST BUCKETS AUTONOMOUSLY**:
   - Strict prohibition against proactively listing, querying, scanning, or enumerating GCS buckets in the project (`gcloud storage ls`, `python3 tools/infrastructure/bucket_manager.py --action=list`, `resolve-existing`, API `storage.buckets.list`, etc.) without explicit user instruction.
   - If a target dataset or bucket path is needed, ALWAYS ask the user directly in the interactive questionnaire or leave it as `[Pending User Provision/Confirmation]` in the execution plan review table.

---

## 📁 MANDATORY RULE 5: MANAGED LUSTRE DATASET DISCOVERY & STAGING PROTOCOL
1. **INSPECTION FIRST**: When benchmarking Managed Lustre, always inspect the Lustre mount path (`/lustre`) to determine if the target dataset shards are already present.
2. **EXPLICIT STAGING NOTIFICATION (WHEN NOT FOUND)**:
   - If the dataset cannot be located on Lustre, the agent MUST explicitly inform the user before copying/staging data from GCS to Lustre.
   - Detail the source GCS path, destination Lustre path, transfer method, estimated transfer size/time, and obtain explicit user confirmation.
3. **DISCOVERY STATUS VISIBILITY (WHEN FOUND)**:
   - If the dataset is discovered pre-existing on Lustre (no copy needed), the agent MUST explicitly report this in the Input Dataset Overview table (e.g. `Discovered Pre-Existing in Lustre — 0 MB copy required, 100% parity`).

---

## 🔀 MANDATORY RULE 6: MANDATORY SHUFFLE STRATEGY VISIBILITY
The active Shuffle Strategy (`none`, `two_stage`, or `global`) MUST be explicitly presented in:
1. The interactive questionnaire alignment.
2. The structured Execution Plan Review table.
3. The final comparative performance summary table.

---

## 🔌 MANDATORY RULE 7: MANDATORY STORAGE ACCESS MODE ALIGNMENT (GCSFUSE VS DIRECT GCS VS LUSTRE)
Whenever benchmarking dataset loading, dataloaders, or storage backends:
1. **ALWAYS ASK & ALIGN ACCESS MODE**: The Agent MUST explicitly ask the user and clarify the Storage Client Access Mode:
   - **GCSFuse CSI Driver Mount (`accessMode=gcsfuse`)**: POSIX filesystem mount with kernel VFS / file cache options.
   - **Direct GCS Client (`accessMode=native_gcs` / `gcsfs`)**: Native Python/C++ HTTP/gRPC Cloud Storage client.
   - **Managed Lustre (`accessMode=lustre`)**: High-performance parallel filesystem mount.
   - **Full Comparison Matrix**: Comparing GCSFuse vs Direct GCS vs Lustre side-by-side.
2. **VISIBILITY REQUIREMENTS**: The active Access Mode and its mount/client configurations MUST be explicitly reported in:
   - The interactive questionnaire alignment.
   - The structured Execution Plan Review table.
   - The final comparative performance summary table.

---

## 🔍 MANDATORY RULE 8: MANDATORY ENVIRONMENT PRE-FLIGHT DIAGNOSTICS & REMEDIATION PLAN PROTOCOL
Before starting any benchmark execution, provisioning resources, or running dataset converters/generators:
1. **ENVIRONMENT & DEPENDENCY CHECK FIRST**: The Agent MUST run `python3 tools/infrastructure/env_checker.py --format=table` to check the user's environment, required CLI tools (`gcloud`, `kubectl`, `helm`, `python3`, `git`), required Python packages (`google-cloud-storage`, `pyyaml`, `pyarrow`, etc.), and GCP/Kubernetes authentication state.
2. **NO AUTONOMOUS DIRECT FIXES**: If any required CLI tool, Python package, or authentication check fails pre-flight verification, the Agent MUST NOT unilaterally execute installation or fix commands (`pip install ...`, `gcloud components install ...`, `gcloud auth login`, etc.).
3. **MANDATORY REMEDIATION PLAN REVIEW**: The Agent MUST present a structured Remediation Plan detailing the missing dependencies and the proposed exact installation commands.
4. **EXPLICIT USER APPROVAL REQUIRED**: The Agent MUST pause execution and wait for explicit user confirmation (e.g. "Proceed" / "确认") before executing any remediation commands.

---

## 📊 MANDATORY RULE 9: STANDARDIZED 5-PART BENCHMARK REPORT STRUCTURE & ANONYMIZATION PROTOCOL
Whenever generating, formatting, or updating benchmark result documents (`docs/**/results/*.md`):
1. **MANDATORY 5-STAGE STRUCTURE**: Every benchmark report MUST strictly adhere to the 5-stage specification:
   - **Section 1: Objective & Evaluation Scope**: Target workload, dataset/model scale, comparison matrix, and key metrics.
   - **Section 2: Testbed Configuration & Workload Dimensions**: Categorized parity table (`Compute & Cluster`, `Storage & CSI`, `Model/Dataset`, `Execution/Methodology`).
   - **Section 3: Empirical Performance Results**: Clean, unified GitHub Markdown comparison table (metric, baseline, optimized, gain/speedup) + optional breakdown sub-tables.
   - **Section 4: Technical Analysis & Key Findings**: High-density numbered deep-dives explaining I/O, compute, network, and storage format mechanisms.
   - **Section 5: Production Recommendations & ROI Analysis / Related Documentation**: Practical selection advice, cost/time ROI break-even analysis, and links to Overview/Reproduction guides.
2. **STRICT ZERO-DUPLICATION POLICY**: Do NOT repeat identical performance numbers across multiple narrative sections. Keep tables central and narratives focused on technical causality.
3. **STRICT PUBLIC ANONYMIZATION**: NEVER expose private cluster names (e.g. `*-gke-persistent`), internal GCP project IDs, personal GCS bucket names, or user emails. Use generic technical specifications (e.g. `Standard GKE Node Pool (n4-standard-80, 80 vCPU, 314 GiB RAM)`).
4. **ALL-ENGLISH TABLES**: All result summary tables, column headers, and metric names MUST be formatted in English.

---

## 🏢 MANDATORY RULE 10: PRIMARY GOOGLE3 WORKSPACE & STRICT NO GITHUB PUSH PROTOCOL
1. **PRIMARY DEFAULT WORKSPACE**: All ongoing development, bug fixes, benchmark suites, documentation, and agent skills MUST default to the internal Google3 CitC workspace at `/google/src/cloud/chongliu/gcloud-ml-benchmarks/google3/experimental/users/chongliu/gcloud_ml_benchmarks/`.
2. **INTERNAL PIPER/G4 SCM**: All code edits, new files, and changelists MUST be managed via Piper / `g4` tools (`g4 change`, `g4 edit`, `g4 add`, `g4 submit`).
3. **STRICT PROHIBITION ON GITHUB PUSH**: Strict prohibition against pushing commits, branches, or PRs to external GitHub (`git push`, `gh pr create`) unless the user gives an explicit, unambiguous command to do so.
