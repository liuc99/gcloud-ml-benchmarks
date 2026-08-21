#!/usr/bin/env python3
"""
Unit tests for tools/checkpoints/orbax_reshard_rewriter.py
Validates CLI parsing, chunk calculation, PyTree scanning, and end-to-end checkpoint resharding.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.checkpoints.orbax_reshard_rewriter import (
    ARRAY_MARKERS,
    compute_target_chunks,
    count_chunks,
    detect_tensorstore_driver,
    parse_args,
    parse_dim_partitions,
    rewrite_array,
    run_orbax_rewrite,
    scan_checkpoint_tree,
    ts,
)


class TestOrbaxReshardRewriter(unittest.TestCase):
    """Unit tests for Orbax checkpoint offline resharder."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_orbax_rewrite_")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_mock_checkpoint(self, base_dir: str):
        """Creates a realistic mock Orbax checkpoint directory structure."""
        if ts is None:
            return

        # 1. Metadata files
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, "commit_success.txt"), "w") as f:
            f.write("committed\n")
        with open(os.path.join(base_dir, "_CHECKPOINT"), "w") as f:
            json.dump({"format": "orbax", "version": 1}, f)
        with open(os.path.join(base_dir, ".orbax-checkpoint-metadata"), "w") as f:
            json.dump({"step": 1000, "timestamp": 1234567890}, f)

        # 2. Params: Layer 0 weight (shape [128, 64], chunks [16, 64] -> 8 chunks)
        l0_w_dir = os.path.join(base_dir, "items", "params", "layer0", "weight")
        l0_w_spec = {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": l0_w_dir},
            "metadata": {"shape": [128, 64], "chunks": [16, 64], "dtype": "<f4"},
            "create": True,
        }
        l0_w_ts = ts.open(l0_w_spec).result()
        l0_w_data = np.arange(128 * 64, dtype=np.float32).reshape(128, 64)
        l0_w_ts.write(l0_w_data).result()

        # 3. Params: Layer 0 bias (shape [64], chunks [16] -> 4 chunks)
        l0_b_dir = os.path.join(base_dir, "items", "params", "layer0", "bias")
        l0_b_spec = {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": l0_b_dir},
            "metadata": {"shape": [64], "chunks": [16], "dtype": "<f4"},
            "create": True,
        }
        l0_b_ts = ts.open(l0_b_spec).result()
        l0_b_ts.write(np.ones(64, dtype=np.float32)).result()

        # 4. Opt state: mu weight (shape [128, 64], chunks [16, 64] -> 8 chunks)
        opt_mu_dir = os.path.join(base_dir, "items", "opt_state", "mu", "layer0", "weight")
        opt_mu_spec = {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": opt_mu_dir},
            "metadata": {"shape": [128, 64], "chunks": [16, 64], "dtype": "<f4"},
            "create": True,
        }
        opt_mu_ts = ts.open(opt_mu_spec).result()
        opt_mu_ts.write(np.zeros((128, 64), dtype=np.float32)).result()

    def test_cli_args_parsing(self):
        test_args = [
            "--src-dir", "/tmp/src_ckpt",
            "--dst-dir", "/tmp/dst_ckpt",
            "--target-chunk-mb", "32.0",
            "--strategy", "optimal_size",
            "--strip-opt-state",
            "--num-workers", "2",
        ]
        with patch("sys.argv", ["orbax_reshard_rewriter.py"] + test_args):
            args = parse_args()
            self.assertEqual(args.src_dir, "/tmp/src_ckpt")
            self.assertEqual(args.dst_dir, "/tmp/dst_ckpt")
            self.assertEqual(args.target_chunk_mb, 32.0)
            self.assertEqual(args.strategy, "optimal_size")
            self.assertTrue(args.strip_opt_state)
            self.assertEqual(args.num_workers, 2)

    def test_chunk_computations(self):
        # Small array: should result in a single unpartitioned chunk
        chunks_small = compute_target_chunks(
            shape=[100, 100],
            dtype_size=4,
            strategy="optimal_size",
            target_chunk_mb=64.0,
            dim_partitions={},
        )
        self.assertEqual(chunks_small, [100, 100])
        self.assertEqual(count_chunks([100, 100], chunks_small), 1)

        # Unsharded strategy
        chunks_unsharded = compute_target_chunks(
            shape=[4096, 4096],
            dtype_size=4,
            strategy="unsharded",
            target_chunk_mb=64.0,
            dim_partitions={},
        )
        self.assertEqual(chunks_unsharded, [4096, 4096])

        # Dimension partitions strategy
        dim_parts = parse_dim_partitions("0:4,1:2")
        chunks_parts = compute_target_chunks(
            shape=[1000, 200],
            dtype_size=4,
            strategy="dim_partitions",
            target_chunk_mb=64.0,
            dim_partitions=dim_parts,
        )
        self.assertEqual(chunks_parts, [250, 100])
        self.assertEqual(count_chunks([1000, 200], chunks_parts), 8)

    @unittest.skipIf(ts is None, "tensorstore is not installed")
    def test_scan_checkpoint_tree(self):
        src_dir = os.path.join(self.test_dir, "src_ckpt")
        self._create_mock_checkpoint(src_dir)

        # Scan all (including opt_state)
        arrs, metas = scan_checkpoint_tree(src_dir, strip_opt_state=False)
        self.assertEqual(len(arrs), 3)  # layer0/weight, layer0/bias, opt_state/mu/layer0/weight
        self.assertGreaterEqual(len(metas), 3)  # commit_success.txt, _CHECKPOINT, .orbax-checkpoint-metadata

        # Scan with strip_opt_state=True
        arrs_stripped, metas_stripped = scan_checkpoint_tree(src_dir, strip_opt_state=True)
        self.assertEqual(len(arrs_stripped), 2)  # layer0/weight, layer0/bias
        for a in arrs_stripped:
            self.assertNotIn("opt_state", a)

    @unittest.skipIf(ts is None, "tensorstore is not installed")
    def test_end_to_end_rewrite_optimal_size(self):
        src_dir = os.path.join(self.test_dir, "src_ckpt")
        dst_dir = os.path.join(self.test_dir, "dst_ckpt")
        self._create_mock_checkpoint(src_dir)

        summary = run_orbax_rewrite(
            src_dir=src_dir,
            dst_dir=dst_dir,
            strategy="optimal_size",
            target_chunk_mb=64.0,
            strip_opt_state=True,
            verify=True,
            dry_run=False,
        )

        self.assertEqual(summary["arrays_processed"], 2)
        self.assertTrue(summary["strip_opt_state"])
        self.assertGreater(summary["source_total_chunks"], summary["target_total_chunks"])
        self.assertGreater(summary["chunk_reduction_percent"], 0.0)

        # Verify destination files
        self.assertTrue(os.path.exists(os.path.join(dst_dir, "commit_success.txt")))
        self.assertTrue(os.path.exists(os.path.join(dst_dir, "_CHECKPOINT")))
        self.assertTrue(os.path.exists(os.path.join(dst_dir, ".orbax-checkpoint-metadata")))
        self.assertTrue(os.path.exists(os.path.join(dst_dir, "rewrite_manifest.json")))
        self.assertFalse(os.path.exists(os.path.join(dst_dir, "items", "opt_state")))

        # Read back rewritten weight
        dst_l0_w_dir = os.path.join(dst_dir, "items", "params", "layer0", "weight")
        dst_ts = ts.open({
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": dst_l0_w_dir},
        }).result()
        self.assertEqual(list(dst_ts.shape), [128, 64])
        self.assertEqual(dst_ts.spec().to_json()["metadata"]["chunks"], [128, 64])

    @unittest.skipIf(ts is None, "tensorstore is not installed")
    def test_end_to_end_rewrite_with_dtype_cast(self):
        src_dir = os.path.join(self.test_dir, "src_ckpt")
        dst_dir = os.path.join(self.test_dir, "dst_ckpt_bf16")
        self._create_mock_checkpoint(src_dir)

        summary = run_orbax_rewrite(
            src_dir=src_dir,
            dst_dir=dst_dir,
            strategy="optimal_size",
            cast_dtype="bfloat16",
            strip_opt_state=False,
            verify=True,
            dry_run=False,
        )

        self.assertEqual(summary["cast_dtype"], "bfloat16")
        dst_l0_w_dir = os.path.join(dst_dir, "items", "params", "layer0", "weight")
        dst_ts = ts.open({
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": dst_l0_w_dir},
        }).result()
        self.assertEqual(dst_ts.dtype.name, "bfloat16")

    @unittest.skipIf(ts is None, "tensorstore is not installed")
    def test_dry_run_mode(self):
        src_dir = os.path.join(self.test_dir, "src_ckpt")
        dst_dir = os.path.join(self.test_dir, "dst_ckpt_dry")
        self._create_mock_checkpoint(src_dir)

        summary = run_orbax_rewrite(
            src_dir=src_dir,
            dst_dir=dst_dir,
            strategy="optimal_size",
            dry_run=True,
        )

        self.assertTrue(summary["dry_run"])
        self.assertFalse(os.path.exists(dst_dir))


if __name__ == "__main__":
    unittest.main()
