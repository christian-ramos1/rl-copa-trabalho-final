"""Machine diagnostics must distinguish host totals from inherited quotas."""
from pathlib import Path

import pytest

from scripts.football_machine import cgroup_nodes, limits
from scripts.football_benchmark import timing, counter_delta


def test_v2_mount_root_and_visible_ancestors():
    mounts = '29 23 0:26 /tenant /sys/fs/cgroup rw - cgroup2 cgroup rw'
    assert cgroup_nodes('0::/tenant/job', mounts) == [Path('/sys/fs/cgroup/job'), Path('/sys/fs/cgroup')]
    assert cgroup_nodes('0::/', mounts) == [Path('/sys/fs/cgroup')]


def test_v1_controller_mount():
    mounts = '29 23 0:26 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct'
    assert cgroup_nodes('2:cpu,cpuacct:/job\n3:memory:/other', mounts) == [Path('/sys/fs/cgroup/cpu/job'), Path('/sys/fs/cgroup/cpu')]


def test_parent_quota_and_memory_headroom_override_child_max():
    rows = [dict(values={'cpu.max':'max 100000', 'memory.max':'max'}),
            dict(values={'cpu.max':'150000 100000', 'memory.max':'1000', 'memory.current':'700'})]
    result = limits(rows, 12)
    assert result['cpu_visible_upper_bound'] == 1.5
    assert result['memory_limit_bytes_min'] == 1000
    assert result['memory_headroom_bytes_min'] == 300


def test_v1_unlimited_memory_is_not_reported_as_available_ram():
    result = limits([dict(values={'cpu.cfs_quota_us':'200000', 'cpu.cfs_period_us':'100000',
                                 'memory.limit_in_bytes':str(2**63-4096)})], 12)
    assert result['cpu_visible_upper_bound'] == 2
    assert result['memory_limit_bytes_min'] is None


def test_timing_excludes_warmup_and_final_evaluation():
    rows = [dict(global_step=512,session_seconds=10),dict(global_step=1024,session_seconds=12)]
    result = timing(rows, 512, 1024)
    assert result == dict(measured_steps=512,training_seconds=2,training_sps=256)
    with pytest.raises(ValueError,match='incomplete'):
        timing(rows,512,2048)


def test_throttling_counter_delta_preserves_ancestor_paths():
    before = dict(cgroups=[dict(path='/cg/job',values={'cpu.stat':'nr_throttled 10\nusage_usec 100'})])
    after = dict(cgroups=[dict(path='/cg/job',values={'cpu.stat':'nr_throttled 14\nusage_usec 900'})])
    assert counter_delta(before,after,'cpu.stat','nr_throttled') == 4
    assert counter_delta(before,after,'memory.events','oom_kill') is None
