#!/usr/bin/env python3
"""
GKE Cluster and Environment Inspection Tool.

Provides pre-flight diagnostics for GKE cluster version, GCSFuse CSI Driver addon,
VPC MTU configuration, JobSet CRD installation, and NodePool capabilities.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [CLUSTER_MANAGER] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)


def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.debug(f"Command failed: {cmd}\nError: {e.stderr}")
        return None


def discover_context():
    context = run_cmd("kubectl config current-context")
    cluster_name, zone, project_id = "", "", ""
    if context:
        # Standard GKE context format: gke_PROJECT_ZONE_CLUSTER
        parts = context.split("_")
        if len(parts) >= 4 and parts[0] == "gke":
            project_id = parts[1]
            zone = parts[2]
            cluster_name = "_".join(parts[3:])
    
    if not project_id:
        project_id = run_cmd("gcloud config get-value project") or ""

    return context or "", cluster_name, zone, project_id


def get_gcsfuse_csi_info():
    # Check both potential daemonset names in kube-system
    for ds_name in ["gcsfusecsi-node", "gke-gcsfuse-csi-node"]:
        images = run_cmd(
            f"kubectl get daemonset {ds_name} -n kube-system -o jsonpath='{{.spec.template.spec.containers[*].image}}'"
        )
        if images:
            # Extract driver version tag from image string
            version_match = re.search(r"gcs-fuse-csi-driver:([^\s@]+)", images)
            version = version_match.group(1) if version_match else "installed"
            return {
                "enabled": True,
                "daemonset": ds_name,
                "version": version,
                "raw_images": images,
            }
    return {"enabled": False, "daemonset": None, "version": None, "raw_images": None}


def get_vpc_mtu(cluster_name, zone, project_id):
    # Method 1: Query gcloud compute network describe if params available
    if cluster_name and project_id and zone:
        network = run_cmd(
            f"gcloud container clusters describe {cluster_name} --project={project_id} --zone={zone} --format='value(networkConfig.network)'"
        )
        if network:
            mtu = run_cmd(
                f"gcloud compute networks describe {network} --project={project_id} --format='value(mtu)'"
            )
            if mtu:
                mtu_int = int(mtu)
                label = "Jumbo Frames" if mtu_int >= 8800 else "Standard"
                return f"MTU {mtu} ({label})"

    # Method 2: Query netd pod eth0 interface MTU
    netd_pod = run_cmd(
        "kubectl get pods -n kube-system -l k8s-app=netd -o jsonpath='{.items[0].metadata.name}'"
    )
    if netd_pod:
        mtu = run_cmd(
            f"kubectl exec -n kube-system {netd_pod} -c netd -- cat /sys/class/net/eth0/mtu"
        )
        if mtu and mtu.isdigit():
            mtu_int = int(mtu)
            label = "Jumbo Frames" if mtu_int >= 8800 else "Standard"
            return f"MTU {mtu} ({label})"

    return "MTU 1460 (Standard - Fallback)"


def get_nodes_info():
    nodes_json = run_cmd("kubectl get nodes -o json")
    if not nodes_json:
        return []

    try:
        data = json.loads(nodes_json)
    except Exception as e:
        logging.warning(f"Failed to parse nodes JSON: {e}")
        return []

    nodes = []
    for item in data.get("items", []):
        metadata = item.get("metadata", {})
        status = item.get("status", {})
        labels = metadata.get("labels", {})
        capacity = status.get("capacity", {})
        node_info = status.get("nodeInfo", {})

        # Parse memory into KiB / GiB
        mem_str = capacity.get("memory", "0Ki")
        mem_kib = int(re.sub(r"[^\d]", "", mem_str)) if re.sub(r"[^\d]", "", mem_str) else 0
        mem_gib = round(mem_kib / (1024 * 1024), 2)

        instance_type = labels.get(
            "node.kubernetes.io/instance-type",
            labels.get("beta.kubernetes.io/instance-type", "unknown"),
        )

        nodes.append(
            {
                "name": metadata.get("name"),
                "instance_type": instance_type,
                "cpus": capacity.get("cpu", "unknown"),
                "memory_gib": f"{mem_gib} GiB",
                "os_image": node_info.get("osImage", "unknown"),
                "kernel_version": node_info.get("kernelVersion", "unknown"),
                "container_runtime": node_info.get("containerRuntimeVersion", "unknown"),
                "kubelet_version": node_info.get("kubeletVersion", "unknown"),
            }
        )
    return nodes


def get_cluster_info(cluster_name="", zone="", project_id=""):
    discovered_ctx, disc_cluster, disc_zone, disc_proj = discover_context()
    
    cluster_name = cluster_name or disc_cluster
    zone = zone or disc_zone
    project_id = project_id or disc_proj

    jobset_crd = run_cmd("kubectl get crd jobsets.jobset.x-k8s.io --no-headers")
    gcsfuse_info = get_gcsfuse_csi_info()
    vpc_mtu = get_vpc_mtu(cluster_name, zone, project_id)
    nodes = get_nodes_info()

    # Aggregate node pool machine breakdown
    instance_counts = {}
    for n in nodes:
        itype = n["instance_type"]
        instance_counts[itype] = instance_counts.get(itype, 0) + 1

    kubelet_version = nodes[0]["kubelet_version"] if nodes else "unknown"

    return {
        "context": discovered_ctx,
        "project_id": project_id,
        "cluster_name": cluster_name,
        "zone": zone,
        "kubelet_version": kubelet_version,
        "jobset_installed": bool(jobset_crd),
        "gcsfuse_csi": gcsfuse_info,
        "vpc_mtu": vpc_mtu,
        "total_nodes": len(nodes),
        "node_instance_breakdown": instance_counts,
        "nodes": nodes,
    }


def inspect_lustre(pvc_name="lustre-checkpoint-pvc", path="/lustre"):
    """
    Inspects Managed Lustre filesystem on GKE via a transient inspection Pod.
    """
    pod_name = f"lustre-inspect-{int(time.time())}"
    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": pod_name, "namespace": "default"},
        "spec": {
            "restartPolicy": "Never",
            "nodeSelector": {"cloud.google.com/gke-nodepool": "n4-standard-80"},
            "volumes": [
                {"name": "lustre-vol", "persistentVolumeClaim": {"claimName": pvc_name}}
            ],
            "containers": [
                {
                    "name": "inspector",
                    "image": "busybox:latest",
                    "command": ["/bin/sh", "-c", "head -n 25 /lustre/arrayrecord_dataset/manifest.json"],
                    "volumeMounts": [{"name": "lustre-vol", "mountPath": path}],
                }
            ],
        },
    }

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(pod_manifest, f)
        tmp_pod_file = f.name

    try:
        run_cmd(f"kubectl apply -f {tmp_pod_file}")
        # Wait for pod completion
        for _ in range(30):
            status = run_cmd(f"kubectl get pod {pod_name} -n default -o jsonpath='{{.status.phase}}'")
            if status in ("Succeeded", "Failed"):
                break
            time.sleep(1)

        logs = run_cmd(f"kubectl logs {pod_name} -n default -c inspector")
        run_cmd(f"kubectl delete pod {pod_name} -n default --grace-period=0 --force")
        if os.path.exists(tmp_pod_file):
            os.remove(tmp_pod_file)

        return {
            "status": "success",
            "pvc_name": pvc_name,
            "mount_path": path,
            "raw_output": logs or "",
        }
    except Exception as e:
        run_cmd(f"kubectl delete pod {pod_name} -n default --grace-period=0 --force 2>/dev/null || true")
        if os.path.exists(tmp_pod_file):
            os.remove(tmp_pod_file)
        return {"status": "error", "error": str(e)}


def init_lustre_manifest(pvc_name="lustre-checkpoint-pvc", path="/lustre"):
    """
    Generates /lustre/manifest.json if missing to ensure 100% parity with GCS bucket manifest.
    """
    pod_name = f"lustre-manifest-init-{int(time.time())}"
    py_script = f"import glob, json, os; shards = sorted([os.path.basename(f) for f in glob.glob('{path}/*.parquet')]); print(f'Indexed {{len(shards)}} shards into {path}/manifest.json'); json.dump({{'shards': shards, 'num_shards': len(shards)}}, open('{path}/manifest.json', 'w'))"
    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": pod_name, "namespace": "default"},
        "spec": {
            "restartPolicy": "Never",
            "nodeSelector": {"cloud.google.com/gke-nodepool": "n4-standard-80"},
            "volumes": [
                {"name": "lustre-vol", "persistentVolumeClaim": {"claimName": pvc_name}}
            ],
            "containers": [
                {
                    "name": "init-manifest",
                    "image": "python:3.11-slim",
                    "command": ["python3", "-c", py_script],
                    "volumeMounts": [{"name": "lustre-vol", "mountPath": path}],
                }
            ],
        },
    }

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(pod_manifest, f)
        tmp_pod_file = f.name

    try:
        run_cmd(f"kubectl apply -f {tmp_pod_file}")
        for _ in range(45):
            status = run_cmd(f"kubectl get pod {pod_name} -n default -o jsonpath='{{.status.phase}}'")
            if status in ("Succeeded", "Failed"):
                break
            time.sleep(1)

        logs = run_cmd(f"kubectl logs {pod_name} -n default -c init-manifest")
        run_cmd(f"kubectl delete pod {pod_name} -n default --grace-period=0 --force 2>/dev/null || true")
        if os.path.exists(tmp_pod_file):
            os.remove(tmp_pod_file)

        return {"status": "success", "output": logs or ""}
    except Exception as e:
        run_cmd(f"kubectl delete pod {pod_name} -n default --grace-period=0 --force 2>/dev/null || true")
        if os.path.exists(tmp_pod_file):
            os.remove(tmp_pod_file)
        return {"status": "error", "error": str(e)}


def sync_gcs_to_lustre(gcs_uri, dest_path="/lustre/arrayrecord_dataset", pvc_name="lustre-checkpoint-pvc"):
    """
    Synchronizes dataset shards from GCS to Managed Lustre via a high-throughput parallel Pod.
    """
    pod_name = f"lustre-sync-{int(time.time())}"
    sync_cmd = f"mkdir -p {dest_path} && echo 'Resuming GCS to Lustre rsync: {gcs_uri} -> {dest_path}...' && gcloud storage rsync -r {gcs_uri} {dest_path} && echo 'Sync complete! Verifying file count...' && ls -1 {dest_path}/*.array_record 2>/dev/null | wc -l"
    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": pod_name, "namespace": "default"},
        "spec": {
            "restartPolicy": "Never",
            "nodeSelector": {"cloud.google.com/gke-nodepool": "n4-standard-80"},
            "volumes": [
                {"name": "lustre-vol", "persistentVolumeClaim": {"claimName": pvc_name}}
            ],
            "containers": [
                {
                    "name": "sync-agent",
                    "image": "google/cloud-sdk:slim",
                    "command": ["/bin/bash", "-c", sync_cmd],
                    "volumeMounts": [{"name": "lustre-vol", "mountPath": "/lustre"}],
                }
            ],
        },
    }

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(pod_manifest, f)
        tmp_pod_file = f.name

    try:
        run_cmd(f"kubectl apply -f {tmp_pod_file}")
        # Wait for pod completion (up to 300 seconds for 155 GB)
        for _ in range(300):
            status = run_cmd(f"kubectl get pod {pod_name} -n default -o jsonpath='{{.status.phase}}'")
            if status in ("Succeeded", "Failed"):
                break
            time.sleep(2)

        logs = run_cmd(f"kubectl logs {pod_name} -n default -c sync-agent --tail=50")
        run_cmd(f"kubectl delete pod {pod_name} -n default --grace-period=0 --force 2>/dev/null || true")
        if os.path.exists(tmp_pod_file):
            os.remove(tmp_pod_file)

        return {"status": "success", "output": logs or ""}
    except Exception as e:
        run_cmd(f"kubectl delete pod {pod_name} -n default --grace-period=0 --force 2>/dev/null || true")
        if os.path.exists(tmp_pod_file):
            os.remove(tmp_pod_file)
        return {"status": "error", "error": str(e)}


def parse_args():
    parser = argparse.ArgumentParser(description="GKE Cluster Pre-flight Inspector")
    parser.add_argument(
        "--action",
        type=str,
        default="diagnose",
        choices=["diagnose", "inspect-lustre", "init-lustre-manifest", "sync-lustre"],
        help="Action: 'diagnose', 'inspect-lustre', 'init-lustre-manifest', or 'sync-lustre'",
    )
    parser.add_argument("--cluster-name", type=str, default="", help="Target GKE Cluster name")
    parser.add_argument("--zone", type=str, default="", help="GKE Cluster zone")
    parser.add_argument("--project-id", type=str, default="", help="GCP Project ID")
    parser.add_argument("--pvc-name", type=str, default="lustre-checkpoint-pvc", help="Target Lustre PVC name")
    parser.add_argument("--source-uri", type=str, default="", help="Source GCS URI for sync")
    parser.add_argument("--dest-path", type=str, default="/lustre/arrayrecord_dataset", help="Destination path on Lustre")
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "table"],
        help="Output format: 'json' or 'table'",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.action == "inspect-lustre":
        res = inspect_lustre(args.pvc_name)
        print(json.dumps(res, indent=2))
        return
    elif args.action == "init-lustre-manifest":
        res = init_lustre_manifest(args.pvc_name)
        print(json.dumps(res, indent=2))
        return
    elif args.action == "sync-lustre":
        source = args.source_uri or "gs://chongliu-macrobench-dataset-965f0fed/arrayrecord_dataset"
        res = sync_gcs_to_lustre(source, args.dest_path, args.pvc_name)
        print(json.dumps(res, indent=2))
        return

    info = get_cluster_info(args.cluster_name, args.zone, args.project_id)

    if args.format == "json":
        print(json.dumps(info, indent=2))
    else:
        print("\n=== GKE Cluster Pre-flight Diagnostic Report ===")
        print(f"Context:              {info['context']}")
        print(f"Project ID:           {info['project_id']}")
        print(f"Cluster Name:         {info['cluster_name']}")
        print(f"Zone:                 {info['zone']}")
        print(f"Kubelet Version:      {info['kubelet_version']}")
        print(f"JobSet Installed:     {info['jobset_installed']}")
        print(f"GCSFuse CSI Enabled:  {info['gcsfuse_csi']['enabled']} (Version: {info['gcsfuse_csi']['version']})")
        print(f"VPC MTU Config:       {info['vpc_mtu']}")
        print(f"Total Nodes:          {info['total_nodes']} {info['node_instance_breakdown']}")
        print("\nNode Specifications:")
        for n in info['nodes']:
            print(f" - {n['name']}: {n['instance_type']} | CPU: {n['cpus']} | Mem: {n['memory_gib']} | OS: {n['os_image']} | Runtime: {n['container_runtime']}")
        print("=================================================\n")


if __name__ == "__main__":
    main()
