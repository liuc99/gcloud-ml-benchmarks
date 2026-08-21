# Agent Behavior Rules & Guardrails (`AGENTS.md`)

## 🛑 MANDATORY RULE 1: SKILL-FIRST PROTOCOL (NEVER FREESTYLE)
Whenever the user proposes or requests running any benchmark, performance test, dataset loading comparison, GCS storage evaluation, or workload execution:
1. **ALWAYS SEARCH & VIEW SKILLS FIRST**: The Agent MUST view and follow the relevant skills in `skills/` (e.g. `skills/ml-benchmark-orchestrator/SKILL.md` and `skills/maxtext-dataset-benchmark/SKILL.md`) using `view_file` BEFORE executing any commands or deployments.
2. **NO AD-HOC INLINE SCRIPTS**: Strict prohibition against executing inline Python (`python3 -c "..."`) or temporary ad-hoc scripts. All cluster diagnostics, bucket management, and dataset inspection MUST use formal committed CLI tools in `tools/` (`tools/infrastructure/cluster_manager.py`, `tools/infrastructure/bucket_manager.py`).

---

## 📋 MANDATORY RULE 2: INTERACTIVE PLAN CONFIRMATION BEFORE EXECUTION
1. **INTERACTIVE ALIGNMENT FIRST**: Always conduct interactive questionnaire alignment to confirm target dataset path, format, shuffle strategies, and access modes.
2. **STRUCTURED PLAN REVIEW TABLE**: Present a structured Markdown table detailing:
   - Workload & Model under test
   - Target GKE cluster and node specs
   - GCSFuse CSI Driver version & VPC MTU
   - Storage backends and configuration flags under test
   - Input dataset overview (total size, shard count, schema)
   - Cloud resource consumption estimate
3. **EXPLICIT USER APPROVAL REQUIRED**: The Agent MUST pause and wait for explicit user confirmation (e.g., "Proceed" / "确认") before invoking sub-skills, creating resources, or running Helm/kubectl deployments.

---

## 🔒 MANDATORY RULE 3: STRICT PERSISTENT RESOURCE & DATASET PROTECTION
1. **IMMUTABLE PERSISTENT ASSETS**: Never delete, overwrite, or mutate pre-existing GCS buckets, persistent GKE clusters, or existing datasets.
2. **DATASET PARITY**: In comparative evaluations, all storage backends/configurations MUST be benchmarked against identical dataset dimensions and schemas (no silent substitution).
3. **MANDATORY COMPLETE POD LIFECYCLE MONITORING**: Never teardown Helm releases while pods are still running. Only teardown ephemeral Helm releases after workload pods reach `Completed` status and logs are captured.
