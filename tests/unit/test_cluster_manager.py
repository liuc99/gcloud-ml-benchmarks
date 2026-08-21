#!/usr/bin/env python3
"""
Unit tests for tools/infrastructure/cluster_manager.py
Validates GKE context discovery, GCSFuse CSI driver detection,
VPC MTU parsing, node spec extraction, and CLI formatting.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.infrastructure.cluster_manager import (
    discover_context,
    get_cluster_info,
    get_gcsfuse_csi_info,
    get_nodes_info,
    get_vpc_mtu,
    parse_args,
)


class TestClusterManager(unittest.TestCase):
    """Tests for cluster manager pre-flight diagnostics."""

    def test_discover_context(self):
        with patch("tools.infrastructure.cluster_manager.run_cmd") as mock_run:
            mock_run.return_value = "gke_test-project-101_us-central1-b_my-persistent-gke"
            ctx, cluster, zone, proj = discover_context()
            self.assertEqual(proj, "test-project-101")
            self.assertEqual(zone, "us-central1-b")
            self.assertEqual(cluster, "my-persistent-gke")

    def test_get_gcsfuse_csi_info_enabled(self):
        with patch("tools.infrastructure.cluster_manager.run_cmd") as mock_run:
            mock_run.side_effect = lambda cmd: (
                "gke.gcr.io/gcs-fuse-csi-driver:v1.22.21-gke.1" if "gcsfusecsi-node" in cmd else None
            )
            info = get_gcsfuse_csi_info()
            self.assertTrue(info["enabled"])
            self.assertEqual(info["version"], "v1.22.21-gke.1")
            self.assertEqual(info["daemonset"], "gcsfusecsi-node")

    def test_get_gcsfuse_csi_info_disabled(self):
        with patch("tools.infrastructure.cluster_manager.run_cmd", return_value=None):
            info = get_gcsfuse_csi_info()
            self.assertFalse(info["enabled"])
            self.assertIsNone(info["version"])

    def test_get_vpc_mtu_jumbo_frames(self):
        with patch("tools.infrastructure.cluster_manager.run_cmd") as mock_run:
            mock_run.side_effect = lambda cmd: (
                "default" if "container clusters describe" in cmd
                else "8896" if "compute networks describe" in cmd
                else None
            )
            mtu_str = get_vpc_mtu("test-cluster", "us-central1-b", "test-project")
            self.assertIn("8896", mtu_str)
            self.assertIn("Jumbo Frames", mtu_str)

    def test_get_nodes_info_parsing(self):
        mock_nodes_payload = json.dumps({
            "items": [
                {
                    "metadata": {
                        "name": "gke-node-1",
                        "labels": {"node.kubernetes.io/instance-type": "n4-standard-80"},
                    },
                    "status": {
                        "capacity": {"cpu": "80", "memory": "329972320Ki"},
                        "nodeInfo": {
                            "osImage": "Container-Optimized OS from Google",
                            "kernelVersion": "6.6.137+",
                            "containerRuntimeVersion": "containerd://2.1.7",
                            "kubeletVersion": "v1.35.6-gke.1049000",
                        },
                    },
                }
            ]
        })

        with patch("tools.infrastructure.cluster_manager.run_cmd", return_value=mock_nodes_payload):
            nodes = get_nodes_info()
            self.assertEqual(len(nodes), 1)
            node = nodes[0]
            self.assertEqual(node["name"], "gke-node-1")
            self.assertEqual(node["instance_type"], "n4-standard-80")
            self.assertEqual(node["cpus"], "80")
            self.assertIn("314.", node["memory_gib"])
            self.assertEqual(node["container_runtime"], "containerd://2.1.7")

    def test_cli_parsing(self):
        test_args = [
            "--cluster-name", "test-cluster",
            "--zone", "us-central1-b",
            "--project-id", "test-proj",
            "--format", "json",
        ]
        with patch("sys.argv", ["cluster_manager.py"] + test_args):
            args = parse_args()
            self.assertEqual(args.cluster_name, "test-cluster")
            self.assertEqual(args.zone, "us-central1-b")
            self.assertEqual(args.project_id, "test-proj")
            self.assertEqual(args.format, "json")


if __name__ == "__main__":
    unittest.main()
