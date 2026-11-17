#!/usr/bin/env python3
"""Inspect visible machine/container limits without changing the environment.

Standard library only. Torch is tested in a child process so unsupported CPU
instructions or an incompatible installation become evidence, not a lost report.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def command(argv, timeout=15):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return dict(returncode=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(returncode=None, error=str(error))


def cgroup_nodes(groups, mounts):
    """Resolve v1/v2 process membership through mount roots and namespaces."""
    memberships = [line.split(":", 2) for line in groups.splitlines() if line.count(":") >= 2]
    nodes = []
    for line in mounts.splitlines():
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        fields, fs = left.split(), right.split()
        if len(fields) < 5 or len(fs) < 3 or fs[0] not in {"cgroup", "cgroup2"}:
            continue
        mount = Path(fields[4].replace("\\040", " "))
        root = PurePosixPath(fields[3].replace("\\040", " "))
        for _, controllers, membership in memberships:
            if fs[0] == "cgroup2" and controllers or fs[0] == "cgroup" and not set(controllers.split(",")) & set(fs[2].split(",")):
                continue
            group = PurePosixPath(membership)
            if group == PurePosixPath("/"):
                path = mount  # namespace-relative root
            else:
                try:
                    path = mount / str(group.relative_to(root))
                except ValueError:
                    path = mount / str(group).lstrip("/")
            if not path.is_relative_to(mount):
                continue
            while True:
                nodes.append(path)
                if path == mount:
                    break
                path = path.parent
    return list(dict.fromkeys(nodes))


CGROUP_FILES = ("cpu.max", "cpu.stat", "cpu.pressure", "cpu.weight", "cpu.shares", "cpuset.cpus.effective",
                "cpuset.cpus", "cpu.cfs_quota_us", "cpu.cfs_period_us",
                "memory.max", "memory.current", "memory.peak", "memory.events",
                "memory.pressure", "memory.high", "memory.swap.max", "memory.swap.current",
                "memory.limit_in_bytes", "memory.usage_in_bytes", "memory.soft_limit_in_bytes")


def limits(rows, affinity_count):
    quotas, memory_limits, headrooms = [], [], []
    for row in rows:
        values = row["values"]
        quota = period = None
        if values.get("cpu.max"):
            q, p = values["cpu.max"].split()
            quota, period = (None if q == "max" else int(q)), int(p)
        elif values.get("cpu.cfs_quota_us"):
            quota = int(values["cpu.cfs_quota_us"])
            period = int(values["cpu.cfs_period_us"])
        if quota is not None and quota > 0 and period and period > 0:
            quotas.append(quota / period)
        raw = values.get("memory.max", values.get("memory.limit_in_bytes"))
        if raw and raw != "max" and 0 < int(raw) < 2**60:
            limit = int(raw)
            memory_limits.append(limit)
            usage = values.get("memory.current", values.get("memory.usage_in_bytes"))
            if usage is not None:
                headrooms.append(max(0, limit - int(usage)))
    return dict(cpu_quota_cores_min=min(quotas) if quotas else None,
                cpu_visible_upper_bound=min([affinity_count, *quotas]) if affinity_count else None,
                memory_limit_bytes_min=min(memory_limits) if memory_limits else None,
                memory_headroom_bytes_min=min(headrooms) if headrooms else None,
                scope="visible ancestors only; null is unknown/no visible finite limit, not dedicated or unlimited hardware")


def resources():
    groups, mounts = read("/proc/self/cgroup") or "", read("/proc/self/mountinfo") or ""
    rows = []
    for path in cgroup_nodes(groups, mounts):
        values = {name: value for name in CGROUP_FILES if (value := read(path / name)) is not None}
        if values:
            rows.append(dict(path=str(path), values=values))
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    shm = shutil.disk_usage("/dev/shm")._asdict() if Path("/dev/shm").exists() else None
    return dict(affinity=affinity, logical_cpus=os.cpu_count(), cgroup_membership=groups,
                cgroups=rows, limits=limits(rows, len(affinity) if affinity is not None else None),
                shm_bytes=shm, meminfo=read("/proc/meminfo"), cpu_pressure=read("/proc/pressure/cpu"))


TORCH_PROBE = '''import json, torch
torch.set_num_threads(1)
x = torch.ones(8, 8, requires_grad=True)
(x @ x).sum().backward()
available, cuda_ok, cuda_error, devices = False, False, None, []
try:
    available = torch.cuda.is_available()
    if available:
        devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        y = torch.ones(8, 8, device="cuda"); y @ y; torch.cuda.synchronize()
        cuda_ok = True
except Exception as error:
    cuda_error = str(error)
print(json.dumps(dict(version=torch.__version__, compiled_cuda=torch.version.cuda,
    cuda_available=available, cpu_test=True, cuda_test=cuda_ok, cuda_error=cuda_error,
    cpu_capability=torch.backends.cpu.get_cpu_capability(), devices=devices)))
'''


def collect(probe_torch=True):
    info = read("/proc/cpuinfo") or ""
    cpu = {key.strip(): value.strip() for line in info.split("\n\n")[0].splitlines()
           if ":" in line for key, value in [line.split(":", 1)]}
    return dict(utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                architecture=platform.machine(), python=sys.version, os_release=read("/etc/os-release"),
                cpu=cpu, resources=resources(), disk_bytes=shutil.disk_usage(Path.cwd())._asdict(),
                docker=command(["docker", "info", "--format", "{{.ServerVersion}} {{.OSType}}/{{.Architecture}}"]),
                gpu=command(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,driver_version", "--format=csv,noheader,nounits"]),
                torch=command([sys.executable, "-c", TORCH_PROBE], timeout=60) if probe_torch else None,
                persistence="not established: confirm provider volume and survival across stop/recreation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    cli = parser.parse_args()
    report = collect()
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    with cli.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"report": str(cli.out), "limits": report["resources"]["limits"],
                      "shm_bytes": report["resources"]["shm_bytes"], "torch": report["torch"]}, indent=2))


if __name__ == "__main__":
    main()
