"""Generate the SYNTHETIC sample cluster logs in data/sample/logs/ (deterministic, seeded).

The output is committed so ingestion, tests and evals are reproducible. Each
template has a fixed count, so questions like "which node had the most OOMKilled
events?" have one known answer that the eval set can check.

    python scripts/generate_sample_logs.py
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "data" / "sample" / "logs"
START = datetime(2026, 10, 14, tzinfo=UTC)
NODES = ["node-1", "node-2", "node-3", "node-4", "node-5"]

# component -> list of (level, message template, count, node weights or None, hour window)
TEMPLATES: dict[str, list[tuple[str, str, int, list[int] | None, tuple[int, int]]]] = {
    "kubelet": [
        ("ERROR", 'Container "api" in pod "shop/checkout-{n}" was OOMKilled (exit code 137), '
                  "memory limit 512Mi", 38, [2, 3, 2, 25, 6], (0, 24)),
        ("ERROR", 'Back-off restarting failed container "worker" in pod "batch/etl-{n}" '
                  "(CrashLoopBackOff)", 25, [3, 12, 3, 4, 3], (0, 24)),
        ("ERROR", "PLEG is not healthy: pleg was last seen active 3m{n}s ago; threshold is 3m0s",
         12, [0, 0, 12, 0, 0], (9, 12)),
        ("ERROR", 'Failed to pull image "registry.example.com/team/app:v{n}": unauthorized: '
                  "authentication required (ImagePullBackOff)", 9, None, (14, 16)),
        ("WARN", "Eviction manager: the node was low on resource: ephemeral-storage (DiskPressure)",
         6, [0, 0, 0, 0, 6], (18, 20)),
        ("ERROR", "Unable to authenticate the request: x509: certificate has expired or is not "
                  "yet valid", 4, [0, 4, 0, 0, 0], (22, 23)),
        ("INFO", 'SyncLoop (PLEG): event for pod "default/web-{n}"', 60, None, (0, 24)),
    ],
    "kube-apiserver": [
        ("ERROR", "etcd request failed: context deadline exceeded (resource=pods verb=LIST "
                  "latency=10.{n}s)", 22, None, (3, 5)),
        ("WARN", 'Slow request: "GET /api/v1/pods" took 4.{n}s (limit not set, no pagination)',
         10, None, (0, 24)),
        ("INFO", 'HTTP "GET /healthz" 200 {n}ms', 40, None, (0, 24)),
    ],
    "etcd": [
        ("ERROR", "etcdserver: mvcc: database space exceeded (quota 2.1 GB, alarm NOSPACE)",
         15, None, (3, 4)),
        ("WARN", "etcdserver: leader changed (term {n})", 8, None, (6, 8)),
        ("WARN", "slow fdatasync: wal sync took 6{n}ms, expected < 10ms", 12, None, (2, 5)),
        ("INFO", "finished scheduled compaction at revision 88{n}", 20, None, (0, 24)),
    ],
    "coredns": [
        ("ERROR", "[ERROR] plugin/errors: 2 api.example.com. A: read udp 10.244.1.{n}:41234->"
                  "8.8.8.8:53: i/o timeout", 18, None, (12, 14)),
        ("INFO", '[INFO] 10.244.2.{n}:51234 - "A IN shop.svc.cluster.local." NOERROR 0.0012s',
         30, None, (0, 24)),
    ],
    "kube-scheduler": [
        ("WARN", 'FailedScheduling pod "ml/train-{n}": 0/5 nodes are available: 3 node(s) had '
                 "untolerated taint {{dedicated: gpu}}", 14, None, (10, 12)),
        ("INFO", 'Successfully bound pod "default/web-{n}" to node', 35, None, (0, 24)),
    ],
    "containerd": [
        ("ERROR", "failed to create shim task: OCI runtime create failed: cgroup driver "
                  "mismatch (container {n})", 10, [0, 10, 0, 0, 0], (1, 3)),
        ("INFO", "StartContainer for container {n} returns successfully", 30, None, (0, 24)),
    ],
    "kube-proxy": [
        ("WARN", "conntrack: failed to clear stale UDP entries for service 10.96.0.{n}",
         5, None, (8, 9)),
        ("INFO", "Syncing ipvs proxier rules took {n}ms", 25, None, (0, 24)),
    ],
}


def main() -> None:
    rng = random.Random(42)
    OUT.mkdir(parents=True, exist_ok=True)
    for component, templates in TEMPLATES.items():
        lines: list[tuple[datetime, str]] = []
        for level, template, count, weights, (h0, h1) in templates:
            for _ in range(count):
                ts = START + timedelta(seconds=rng.uniform(h0 * 3600, h1 * 3600 - 1))
                node = rng.choices(NODES, weights=weights)[0] if weights else rng.choice(NODES)
                msg = template.format(n=rng.randint(1, 99))
                stamp = ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")
                lines.append((ts, f"{stamp} {level} {component} {node} {msg}"))
        lines.sort()
        (OUT / f"{component}.log").write_text("\n".join(line for _, line in lines) + "\n")
        print(f"wrote {len(lines):4d} lines to {component}.log")


if __name__ == "__main__":
    main()
