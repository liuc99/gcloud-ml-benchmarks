---
name: benchmark-metrics-parser
description: Sub-skill for parsing container stdout benchmark logs, extracting dataset load times, raw network throughput, and checkpoint durations, calculating multi-run statistical averages, and generating comparative Markdown reports.
---

# Benchmark Metrics Parser Sub-Skill (`benchmark-metrics-parser`)

This sub-skill parses container benchmark logs, extracts empirical throughput/latency metrics, calculates statistical metrics across multiple repeat runs (Mean, Min, Max, Standard Deviation), and formats comparative Markdown performance reports.

---

## 🛠️ Tasks & Capabilities

### 1. Extract Workload Container Output
Run `kubectl logs` to fetch stdout from the benchmark container:
```bash
kubectl logs job/${RELEASE_NAME}-workload-0 -c workload
```

### 2. Parse Metrics & Calculate Multi-Run Statistics
Pass raw single or JSON array multi-run log outputs to `parse_metrics.py`:
```bash
python3 skills/benchmark-metrics-parser/scripts/parse_metrics.py <log_file_or_json>
```

The script returns single-run metrics or aggregated statistical metrics:
```json
{
  "individual_runs": [ ... ],
  "aggregated_summary": {
    "dataset_load_time_sec": {"mean": 0.28, "min": 0.26, "max": 0.31, "stdev": 0.02},
    "raw_write_speed_mbs": {"mean": 611.51, "min": 595.20, "max": 625.80, "stdev": 12.30},
    "checkpoint_duration_sec": {"mean": 76.80, "min": 75.14, "max": 79.20, "stdev": 1.80},
    "aggregated_throughput_mbs": {"mean": 503.29, "min": 490.10, "max": 515.00, "stdev": 10.50}
  }
}
```

### 3. Format Comparative Matrix Report
Format aggregated results across storage backends into a clean, GitHub-style Markdown comparison table:

### 📊 Comparative Benchmark Matrix Summary (${ITERATIONS} Iterations Averaged)

| Storage Backend | Dataset Load Time (Avg) | Raw Network Write Speed (Avg) | Checkpoint Duration (Avg) | Aggregated Throughput (Avg) | Speedup vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Google Cloud Managed Lustre** | `${LUSTRE_LOAD_AVG}s` | `${LUSTRE_WRITE_AVG} MB/s` | `${LUSTRE_CKPT_AVG}s ± ${LUSTRE_CKPT_STDEV}s` | `${LUSTRE_AGG_AVG} MB/s` | **${LUSTRE_SPEEDUP}x** |
| **GCSFuse (Streaming Writes)** | `${GCSFUSE_LOAD_AVG}s` | `${GCSFUSE_WRITE_AVG} MB/s` | `${GCSFUSE_CKPT_AVG}s ± ${GCSFUSE_CKPT_STDEV}s` | `${GCSFUSE_AGG_AVG} MB/s` | **${GCSFUSE_SPEEDUP}x** |
| **Direct GCS (`gcsfs`)** | `${GCSFS_LOAD_AVG}s` | `${GCSFS_WRITE_AVG} MB/s` | `${GCSFS_CKPT_AVG}s ± ${GCSFS_CKPT_STDEV}s` | `${GCSFS_AGG_AVG} MB/s` | **1.00x (Baseline)** |
