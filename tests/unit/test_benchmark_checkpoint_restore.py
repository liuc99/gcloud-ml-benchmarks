#!/usr/bin/env python3
"""
Unit tests for tools/checkpoints/benchmark_checkpoint_restore.py
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.checkpoints.benchmark_checkpoint_restore import (
    execute_comparison_benchmark,
    generate_synthetic_fsdp_checkpoint,
    parse_args,
    ts,
)


class TestBenchmarkCheckpointRestore(unittest.TestCase):
    """Unit tests for FSDP checkpoint restore benchmark tool."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_bench_restore_")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_cli_args(self):
        test_args = [
            "--src-shards", "4",
            "--dst-workers", "8",
            "--num-layers", "2",
            "--hidden-dim", "256",
            "--num-runs", "2",
        ]
        with patch("sys.argv", ["benchmark_checkpoint_restore.py"] + test_args):
            args = parse_args()
            self.assertEqual(args.src_shards, 4)
            self.assertEqual(args.dst_workers, 8)
            self.assertEqual(args.num_layers, 2)
            self.assertEqual(args.hidden_dim, 256)
            self.assertEqual(args.num_runs, 2)

    @unittest.skipIf(ts is None, "tensorstore is not installed")
    def test_synthetic_checkpoint_generation(self):
        ckpt_dir = os.path.join(self.test_dir, "synth_ckpt")
        total_arrays, total_bytes = generate_synthetic_fsdp_checkpoint(
            ckpt_dir=ckpt_dir,
            num_shards=4,
            num_layers=2,
            hidden_dim=256,
        )
        self.assertEqual(total_arrays, 2 * 7)  # 2 layers * 7 matrices
        self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "commit_success.txt")))
        self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "_CHECKPOINT")))

    @unittest.skipIf(ts is None, "tensorstore is not installed")
    def test_execute_comparison_benchmark(self):
        results = execute_comparison_benchmark(
            work_dir=self.test_dir,
            src_shards=2,
            dst_workers=4,
            num_layers=1,
            hidden_dim=256,
            num_runs=2,
        )
        self.assertIn("unrewritten", results)
        self.assertIn("rewritten", results)
        self.assertIn("comparison", results)
        self.assertGreater(results["comparison"]["speedup_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
