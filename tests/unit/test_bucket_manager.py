#!/usr/bin/env python3
"""
Unit tests for tools/infrastructure/bucket_manager.py
Validates GCS bucket ensure, describe, dataset inspection, manifest generation,
and CLI flag handling using mock GCS client (zero external network / credentials).
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.infrastructure.bucket_manager import (
    describe_bucket,
    ensure_bucket,
    inspect_dataset,
    parse_args,
)


class TestBucketManager(unittest.TestCase):
    """Tests for bucket manager with mocked GCS SDK."""

    def test_describe_bucket_props(self):
        mock_bucket = MagicMock()
        mock_bucket.name = "test-bucket"
        mock_bucket.storage_class = "STANDARD"
        mock_bucket.location = "US-CENTRAL1"
        mock_bucket._properties = {
            "locationType": "region",
            "customPlacementConfig": {},
            "hierarchicalNamespace": {"enabled": False},
            "iamConfiguration": {"uniformBucketLevelAccess": {"enabled": True}},
        }

        desc = describe_bucket(mock_bucket)
        self.assertEqual(desc["name"], "test-bucket")
        self.assertEqual(desc["gs_url"], "gs://test-bucket")
        self.assertEqual(desc["storage_class"], "STANDARD")
        self.assertEqual(desc["location"], "US-CENTRAL1")
        self.assertTrue(desc["uniform_bucket_level_access"])
        self.assertFalse(desc["hns_enabled"])

    def test_ensure_bucket_existing(self):
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.name = "existing-bucket"
        mock_bucket.storage_class = "STANDARD"
        mock_bucket.location = "US-CENTRAL1"
        mock_bucket._properties = {
            "locationType": "region",
            "customPlacementConfig": {},
            "hierarchicalNamespace": {"enabled": False},
            "iamConfiguration": {"uniformBucketLevelAccess": {"enabled": True}},
        }
        mock_client.get_bucket.return_value = mock_bucket

        res = ensure_bucket(
            client=mock_client,
            project_id="test-project",
            bucket_name="existing-bucket",
            bucket_type="regional",
            location="us-central1",
            zone="us-central1-b",
        )

        self.assertEqual(res["name"], "existing-bucket")
        mock_client.get_bucket.assert_called_once_with("existing-bucket")
        mock_client.create_bucket.assert_not_called()

    def test_ensure_rapid_zonal_bucket_creation(self):
        from google.api_core import exceptions

        mock_client = MagicMock()
        mock_client.get_bucket.side_effect = exceptions.NotFound("Bucket not found")

        mock_new_bucket = MagicMock()
        mock_client.bucket.return_value = mock_new_bucket

        mock_created_bucket = MagicMock()
        mock_created_bucket.name = "new-rapid-bucket"
        mock_created_bucket.storage_class = "RAPID"
        mock_created_bucket.location = "US-CENTRAL1"
        mock_created_bucket._properties = {
            "locationType": "zone",
            "customPlacementConfig": {"dataLocations": ["US-CENTRAL1-B"]},
            "hierarchicalNamespace": {"enabled": True},
            "iamConfiguration": {"uniformBucketLevelAccess": {"enabled": True}},
        }
        mock_client.create_bucket.return_value = mock_created_bucket

        res = ensure_bucket(
            client=mock_client,
            project_id="test-project",
            bucket_name="new-rapid-bucket",
            bucket_type="zonal",
            location="us-central1",
            zone="us-central1-b",
        )

        self.assertEqual(res["name"], "new-rapid-bucket")
        self.assertEqual(res["storage_class"], "RAPID")
        self.assertTrue(res["hns_enabled"])
        self.assertEqual(mock_new_bucket.storage_class, "RAPID")
        self.assertEqual(mock_new_bucket.custom_placement_config, {"dataLocations": ["US-CENTRAL1-B"]})
        mock_client.create_bucket.assert_called_once()

    def test_inspect_dataset_parquet(self):
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.storage_class = "STANDARD"
        mock_bucket.location = "US-CENTRAL1"
        mock_client.get_bucket.return_value = mock_bucket

        # Create mock blobs
        blob1 = MagicMock()
        blob1.name = "dataset/shard_00000.parquet"
        blob1.size = 100 * 1024 * 1024  # 100 MB

        blob2 = MagicMock()
        blob2.name = "dataset/shard_00001.parquet"
        blob2.size = 100 * 1024 * 1024  # 100 MB

        mock_bucket.list_blobs.return_value = [blob1, blob2]

        res = inspect_dataset(
            client=mock_client,
            dataset_uri="gs://my-test-bucket/dataset",
            target_format="parquet",
        )

        self.assertEqual(res["bucket_name"], "my-test-bucket")
        self.assertEqual(res["total_shards"], 2)
        self.assertEqual(res["dataset_format"], "Parquet")
        self.assertEqual(res["total_size_mb"], 200.0)

    def test_cli_parsing(self):
        test_args = [
            "--action", "ensure",
            "--project-id", "test-proj",
            "--bucket-name", "gs://test-bucket",
            "--bucket-type", "regional",
            "--location", "us-central1",
        ]
        with patch("sys.argv", ["bucket_manager.py"] + test_args):
            args = parse_args()
            self.assertEqual(args.action, "ensure")
            self.assertEqual(args.project_id, "test-proj")
            self.assertEqual(args.bucket_name, "gs://test-bucket")
            self.assertEqual(args.bucket_type, "regional")


if __name__ == "__main__":
    unittest.main()
