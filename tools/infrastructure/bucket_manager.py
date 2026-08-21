#!/usr/bin/env python3
"""
GCS Bucket Provisioning, Discovery, and Dataset Inspection Tool.

Provides a deterministic CLI tool using ADC credentials (`google.cloud.storage`)
to create, inspect, list, and resolve Regional, Zonal (RAPID), and Hierarchical Namespace (HNS) GCS buckets,
as well as inspecting dataset specs (shard count, dataset size, format) inside buckets.
"""

import argparse
import json
import logging
import os
import sys
from google.cloud import storage
from google.api_core import exceptions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [BUCKET_MANAGER] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)


def parse_args():
    parser = argparse.ArgumentParser(description="GCS Bucket Provisioner and Inspector")
    parser.add_argument(
        "--action",
        type=str,
        default="ensure",
        choices=["ensure", "create", "describe", "list", "delete", "cleanup-prefix", "inspect-dataset", "resolve-existing", "generate-manifest", "cat-object"],
        help="Action: 'ensure', 'create', 'describe', 'list', 'delete', 'cleanup-prefix', 'inspect-dataset', 'resolve-existing', 'generate-manifest', 'cat-object'",
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default="",
        help="GCP Project ID (optional, auto-discovers if omitted)",
    )
    parser.add_argument(
        "--bucket-name",
        type=str,
        default="",
        help="Target bucket name or GCS URI (gs://...)",
    )
    parser.add_argument(
        "--dataset-uri",
        type=str,
        default="",
        help="Target dataset GCS URI (e.g. gs://my-bucket/dataset_dir) for inspect-dataset action",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix path for cleanup-prefix or dataset inspection",
    )
    parser.add_argument(
        "--bucket-type",
        type=str,
        default="regional",
        choices=["regional", "zonal", "hns", "standard"],
        help="Bucket storage type: 'regional', 'zonal' (RAPID), 'hns', or 'standard'",
    )
    parser.add_argument(
        "--location",
        type=str,
        default="us-central1",
        help="GCP region (e.g. us-central1)",
    )
    parser.add_argument(
        "--zone",
        type=str,
        default="us-central1-b",
        help="GCP zone for Zonal RAPID buckets (e.g. us-central1-b)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="",
        choices=["", "parquet", "arrayrecord", "tar", "all"],
        help="Specific dataset format to filter for inspect-dataset ('parquet', 'arrayrecord', 'tar', or empty for auto-detect)",
    )
    parser.add_argument(
        "--filter-name",
        type=str,
        default="",
        help="Sub-string filter for bucket name in 'list' action",
    )
    return parser.parse_args()


def describe_bucket(bucket_obj):
    props = bucket_obj._properties
    return {
        "name": bucket_obj.name,
        "gs_url": f"gs://{bucket_obj.name}",
        "storage_class": bucket_obj.storage_class,
        "location": bucket_obj.location,
        "location_type": props.get("locationType", "unknown"),
        "custom_placement": props.get("customPlacementConfig", {}),
        "hns_enabled": props.get("hierarchicalNamespace", {}).get("enabled", False),
        "uniform_bucket_level_access": props.get("iamConfiguration", {})
        .get("uniformBucketLevelAccess", {})
        .get("enabled", False),
    }


def find_matching_bucket(client, bucket_type, location, zone, max_results=50):
    target_zone_upper = zone.upper()
    target_loc_upper = location.upper()

    count = 0
    for b_summary in client.list_buckets(max_results=max_results):
        try:
            b = client.get_bucket(b_summary.name)
            sc = (b.storage_class or "").upper()
            loc = (b.location or "").upper()
            props = b._properties
            loc_type = props.get("locationType", "").lower()
            data_locs = [
                dl.upper()
                for dl in props.get("customPlacementConfig", {}).get("dataLocations", [])
            ]

            if bucket_type in ("zonal", "rapid"):
                if sc == "RAPID" and (target_zone_upper in data_locs or loc_type == "zone"):
                    return b
            elif bucket_type == "regional":
                if loc == target_loc_upper and loc_type == "region":
                    return b
        except Exception:
            continue
        count += 1
    return None


def ensure_bucket(client, project_id, bucket_name, bucket_type, location, zone):
    if not bucket_name:
        bucket_name = f"maxtext-{bucket_type}-{project_id}"

    clean_name = bucket_name.replace("gs://", "").strip("/")

    # Check if bucket already exists
    try:
        b = client.get_bucket(clean_name)
        logging.info(f"Existing bucket found: gs://{clean_name}")
        return describe_bucket(b)
    except exceptions.NotFound:
        pass
    except Exception as e:
        logging.warning(f"Error describing gs://{clean_name}: {e}")

    # Attempt to create bucket
    bucket = client.bucket(clean_name)
    bucket.iam_configuration.uniform_bucket_level_access_enabled = True

    if bucket_type in ("zonal", "rapid"):
        bucket.storage_class = "RAPID"
        zone_upper = zone.upper()
        bucket.custom_placement_config = {"dataLocations": [zone_upper]}
        bucket.hierarchical_namespace_enabled = True
        create_loc = location.upper()
    elif bucket_type == "hns":
        bucket.storage_class = "STANDARD"
        bucket.hierarchical_namespace_enabled = True
        create_loc = location.upper()
    else:  # regional / standard
        bucket.storage_class = "STANDARD"
        create_loc = location.upper()

    try:
        created_b = client.create_bucket(bucket, location=create_loc)
        logging.info(f"Successfully created {bucket_type} bucket: gs://{clean_name}")
        return describe_bucket(created_b)
    except exceptions.BadRequest as e:
        logging.error(f"Failed to create {bucket_type} bucket gs://{clean_name}: {e}")
        raise e


def delete_bucket(client, bucket_name):
    clean_name = bucket_name.replace("gs://", "").strip("/")
    try:
        b = client.get_bucket(clean_name)
        blobs = list(b.list_blobs())
        logging.info(f"Deleting {len(blobs)} objects from gs://{clean_name}...")
        for blob in blobs:
            blob.delete()
        b.delete()
        logging.info(f"Successfully deleted bucket gs://{clean_name}")
        return {"status": "deleted", "bucket_name": clean_name, "deleted_objects": len(blobs)}
    except exceptions.NotFound:
        logging.info(f"Bucket gs://{clean_name} not found, nothing to delete.")
        return {"status": "not_found", "bucket_name": clean_name}


def cleanup_prefix(client, bucket_name, prefix):
    clean_name = bucket_name.replace("gs://", "").strip("/")
    prefix = prefix.lstrip("/")
    b = client.get_bucket(clean_name)
    blobs = list(b.list_blobs(prefix=prefix))
    logging.info(f"Deleting {len(blobs)} objects matching prefix '{prefix}' from gs://{clean_name}...")
    for blob in blobs:
        blob.delete()
    logging.info(f"Successfully cleaned prefix '{prefix}' from gs://{clean_name}")
    return {"status": "cleaned", "bucket_name": clean_name, "prefix": prefix, "deleted_objects": len(blobs)}


def inspect_dataset(client, dataset_uri, prefix="", target_format=""):
    target_path = dataset_uri or prefix
    
    DATASET_EXTS = {
        "parquet": [".parquet"],
        "arrayrecord": [".array_record", ".arrayrecord"],
        "tar": [".tar", ".tgz", ".tar.gz"],
        "arrow": [".arrow", ".feather"],
        "jsonl": [".jsonl", ".json"],
    }
    CHECKPOINT_EXTS = {".ckpt", ".pt", ".bin", ".safetensors", ".part", ".tmp", ".temp", ".partial"}

    file_entries = [] # list of (name, size, ext)
    storage_class = "POSIX"
    location = "LOCAL"
    bucket_name = ""

    # Check if target is POSIX (Lustre, GCSFuse mount, or local path)
    if not target_path.startswith("gs://") and os.path.exists(target_path):
        clean_uri = target_path
        storage_class = "MANAGED_LUSTRE" if "/lustre" in target_path else ("GCSFUSE" if "/gcs" in target_path else "POSIX")
        location = "LOCAL_PVC"
        for root, dirs, files in os.walk(target_path):
            for f in files:
                full_p = os.path.join(root, f)
                if not os.path.islink(full_p):
                    try:
                        sz = os.path.getsize(full_p)
                        ext = os.path.splitext(f)[1].lower()
                        file_entries.append((full_p, sz, ext))
                    except Exception:
                        pass
    else:
        # GCS URI
        clean_uri = target_path.replace("gs://", "").strip("/")
        parts = clean_uri.split("/", 1)
        bucket_name = parts[0]
        path_prefix = parts[1] if len(parts) > 1 else ""

        try:
            bucket = client.get_bucket(bucket_name)
            storage_class = bucket.storage_class or "STANDARD"
            location = bucket.location or "US"
        except exceptions.NotFound:
            return {"error": f"Bucket gs://{bucket_name} not found", "dataset_uri": f"gs://{clean_uri}"}

        blobs = list(bucket.list_blobs(prefix=path_prefix))
        for b in blobs:
            if not b.name.endswith("/"):
                ext = os.path.splitext(b.name)[1].lower()
                file_entries.append((b.name, b.size or 0, ext))

    # Extension breakdown
    ext_stats = {}
    for name, size, ext in file_entries:
        if ext not in ext_stats:
            ext_stats[ext] = {"count": 0, "size_bytes": 0}
        ext_stats[ext]["count"] += 1
        ext_stats[ext]["size_bytes"] += size

    ext_breakdown = {}
    for ext, st in ext_stats.items():
        ext_breakdown[ext or "(no ext)"] = {
            "count": st["count"],
            "size_mb": round(st["size_bytes"] / (1024 * 1024), 2),
            "size_gb": round(st["size_bytes"] / (1024 * 1024 * 1024), 2),
        }

    # Determine target dataset extension filter
    selected_exts = set()
    if target_format:
        fmt_key = target_format.lower().replace("-", "").replace("_", "")
        if fmt_key in DATASET_EXTS:
            selected_exts = set(DATASET_EXTS[fmt_key])
        elif fmt_key != "all":
            selected_exts = {f".{fmt_key}" if not fmt_key.startswith(".") else fmt_key}

    # Auto-detect primary dataset format if not explicitly forced
    if not selected_exts:
        # Find if known dataset formats exist
        for fmt, exts in DATASET_EXTS.items():
            if any(ext in ext_stats for ext in exts):
                selected_exts = set(exts)
                break

    # If dataset extensions found, isolate dataset shards vs checkpoints / other files
    if selected_exts:
        dataset_files = [e for e in file_entries if e[2] in selected_exts]
        other_files = [e for e in file_entries if e[2] not in selected_exts]
    else:
        # Exclude known checkpoint/temporary files
        dataset_files = [e for e in file_entries if e[2] not in CHECKPOINT_EXTS and not e[0].endswith(".txt") and not e[0].endswith(".log")]
        other_files = [e for e in file_entries if e not in dataset_files]

    total_dataset_bytes = sum(e[1] for e in dataset_files)
    total_mb = round(total_dataset_bytes / (1024 * 1024), 2)
    total_gb = round(total_dataset_bytes / (1024 * 1024 * 1024), 2)
    avg_size_mb = round(total_mb / len(dataset_files), 2) if dataset_files else 0

    detected_format = "Unknown"
    all_dataset_exts = {e[2] for e in dataset_files}
    if ".parquet" in all_dataset_exts:
        detected_format = "Parquet"
    elif ".array_record" in all_dataset_exts or ".arrayrecord" in all_dataset_exts:
        detected_format = "ArrayRecord"
    elif any(ext in all_dataset_exts for ext in [".tar", ".tgz"]):
        detected_format = "WebDataset TAR"

    res = {
        "dataset_uri": clean_uri if clean_uri.startswith("/") else f"gs://{clean_uri}",
        "bucket_name": bucket_name,
        "storage_class": storage_class,
        "location": location,
        "total_shards": len(dataset_files),
        "total_size_mb": total_mb,
        "total_size_gb": total_gb,
        "average_shard_size_mb": avg_size_mb,
        "dataset_format": detected_format,
        "detected_extensions": list(ext_stats.keys()),
        "extension_breakdown": ext_breakdown,
        "sample_files": [e[0] for e in dataset_files[:10]],
        "subfolder_counts": {
            os.path.dirname(e[0]) or "(root)": sum(1 for x in dataset_files if (os.path.dirname(x[0]) or "(root)") == (os.path.dirname(e[0]) or "(root)"))
            for e in dataset_files
        },
    }

    if other_files:
        other_bytes = sum(e[1] for e in other_files)
        res["non_dataset_artifacts"] = {
            "file_count": len(other_files),
            "total_size_gb": round(other_bytes / (1024 * 1024 * 1024), 2),
            "extensions": list({e[2] for e in other_files}),
        }

    return res


def generate_manifest(client, dataset_uri: str, target_format: str = ""):
    """
    Generates and uploads manifest.json into the target dataset directory.
    """
    inspect_res = inspect_dataset(client, dataset_uri, target_format=target_format)
    clean_uri = dataset_uri.replace("gs://", "").strip("/")
    parts = clean_uri.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket, prefix=prefix)
    shard_names = []
    total_bytes = 0

    ext_target = f".{target_format.lower()}" if target_format else ""
    if ext_target == ".arrayrecord":
        ext_target = ".array_record"

    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        rel_path = blob.name[len(prefix):].lstrip("/") if prefix else blob.name
        if rel_path == "manifest.json" or rel_path.endswith("/manifest.json"):
            continue
        # Avoid traversing into subdirectories if indexing a specific prefix/root
        if "/" in rel_path:
            continue
        if ext_target:
            if rel_path.endswith(ext_target):
                shard_names.append(rel_path)
                total_bytes += blob.size
        elif rel_path.endswith(".array_record") or rel_path.endswith(".parquet"):
            shard_names.append(rel_path)
            total_bytes += blob.size

    shard_names = sorted(shard_names)
    fmt = target_format.lower() if target_format else ("arrayrecord" if any(s.endswith(".array_record") for s in shard_names) else "parquet")
    manifest_data = {
        "dataset_format": fmt,
        "num_shards": len(shard_names),
        "total_bytes": total_bytes,
        "shards": shard_names,
    }

    import tempfile
    manifest_blob_name = f"{prefix}/manifest.json".lstrip("/")
    tmp_path = os.path.join(tempfile.gettempdir(), "manifest.json")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    target_uri = f"gs://{bucket_name}/{manifest_blob_name}"
    uploaded = False
    try:
        blob = bucket.blob(manifest_blob_name)
        blob.upload_from_filename(tmp_path)
        uploaded = True
    except Exception as e:
        logging.warning(f"Standard REST upload failed ({e}), trying gcsfs...")

    if not uploaded:
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            with fs.open(target_uri, "w") as f:
                json.dump(manifest_data, f, indent=2)
            uploaded = True
            logging.info(f"Successfully uploaded manifest via gcsfs to {target_uri}")
        except Exception as e:
            logging.warning(f"gcsfs upload failed ({e}), trying gcloud storage cp...")

    if not uploaded:
        import subprocess
        env = dict(os.environ)
        env.pop("CLOUDSDK_AUTH_AUTHORIZATION_TOKEN_FILE", None)
        subprocess.run(["gcloud", "storage", "cp", tmp_path, target_uri], env=env, check=True)

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    return {
        "status": "success",
        "manifest_uri": target_uri,
        "num_shards": len(shard_names),
        "total_bytes": total_bytes,
    }


def cat_object(client, object_uri: str):
    clean_uri = object_uri.replace("gs://", "").strip("/")
    parts = clean_uri.split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""

    bucket = client.bucket(bucket_name)
    blob = bucket.get_blob(blob_name)
    
    # Also check via gcsfs
    gcsfs_found = False
    gcsfs_content = ""
    try:
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        uri_full = object_uri if object_uri.startswith("gs://") else f"gs://{clean_uri}"
        if fs.exists(uri_full):
            gcsfs_found = True
            with fs.open(uri_full, "r") as f:
                gcsfs_content = f.read()
    except Exception as e:
        logging.warning(f"gcsfs check: {e}")

        try:
            txt = blob.download_as_text()
            parsed_info = {}
            try:
                j = json.loads(txt)
                if isinstance(j, dict):
                    parsed_info = {
                        "dataset_format": j.get("dataset_format"),
                        "num_shards": j.get("num_shards", len(j.get("shards", []))),
                        "total_bytes": j.get("total_bytes"),
                        "first_5_shards": j.get("shards", [])[:5],
                        "last_5_shards": j.get("shards", [])[-5:],
                    }
            except Exception:
                pass
            return {
                "status": "found",
                "uri": f"gs://{clean_uri}",
                "size_bytes": blob.size,
                "created": str(blob.time_created),
                "updated": str(blob.updated),
                "storage_class": blob.storage_class,
                "parsed_manifest": parsed_info,
                "preview": txt[:300],
            }
        except Exception:
            pass

    if gcsfs_found:
        return {
            "status": "found",
            "uri": f"gs://{clean_uri}",
            "size_bytes": len(gcsfs_content),
            "source": "gcsfs",
            "preview": gcsfs_content[:500],
        }

    return {"status": "not_found", "uri": f"gs://{clean_uri}"}


def main():
    args = parse_args()

    project_id = args.project_id
    if not project_id:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "")

    client = storage.Client(project=project_id) if project_id else storage.Client()

    if args.action == "cat-object":
        res = cat_object(client, args.dataset_uri or args.bucket_name)
        print(json.dumps(res, indent=2))
    elif args.action in ("ensure", "create"):
        info = ensure_bucket(
            client,
            project_id or client.project,
            args.bucket_name,
            args.bucket_type,
            args.location,
            args.zone,
        )
        print(json.dumps(info, indent=2))
    elif args.action == "describe":
        b = client.get_bucket(args.bucket_name.replace("gs://", "").strip("/"))
        print(json.dumps(describe_bucket(b), indent=2))
    elif args.action == "list":
        buckets = client.list_buckets()
        results = []
        for b in buckets:
            if args.filter_name and args.filter_name.lower() not in b.name.lower():
                continue
            try:
                results.append(describe_bucket(client.get_bucket(b.name)))
            except Exception as e:
                logging.warning(f"Could not describe bucket {b.name}: {e}")
        print(json.dumps(results, indent=2))
    elif args.action == "delete":
        res = delete_bucket(client, args.bucket_name)
        print(json.dumps(res, indent=2))
    elif args.action == "cleanup-prefix":
        res = cleanup_prefix(client, args.bucket_name, args.prefix)
        print(json.dumps(res, indent=2))
    elif args.action == "inspect-dataset":
        res = inspect_dataset(client, args.dataset_uri, args.prefix, target_format=args.format)
        print(json.dumps(res, indent=2))
    elif args.action == "generate-manifest":
        res = generate_manifest(client, args.dataset_uri, target_format=args.format)
        print(json.dumps(res, indent=2))
    elif args.action == "resolve-existing":
        found_b = find_matching_bucket(client, args.bucket_type, args.location, args.zone)
        if found_b:
            print(json.dumps(describe_bucket(found_b), indent=2))
        else:
            print(json.dumps({"error": f"No matching {args.bucket_type} bucket found in project"}, indent=2))
            sys.exit(1)


if __name__ == "__main__":
    main()
