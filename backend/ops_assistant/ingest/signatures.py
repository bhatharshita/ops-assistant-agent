"""Dictionaries used to pull entities out of free text: components and error signatures.

Rule-based extraction is deliberately simple: it is fast, free, explainable and
easy to test. The README discusses swapping in LLM-based entity extraction.
"""

from __future__ import annotations

import re

# canonical component -> aliases that may appear in text or GitHub labels
COMPONENTS: dict[str, list[str]] = {
    "kubelet": ["kubelet", "area/kubelet", "sig/node"],
    "kube-apiserver": ["kube-apiserver", "apiserver", "api server", "area/apiserver"],
    "etcd": ["etcd", "area/etcd", "etcdserver"],
    "kube-scheduler": ["kube-scheduler", "scheduler", "sig/scheduling"],
    "kube-proxy": ["kube-proxy", "area/kube-proxy", "ipvs", "iptables mode"],
    "coredns": ["coredns", "area/dns", "kube-dns", "dns"],
    "containerd": ["containerd", "container runtime", "runc", "area/cri"],
    "metrics-server": ["metrics-server"],
    "ingress-nginx": ["ingress-nginx", "ingress controller"],
    "kube-state-metrics": ["kube-state-metrics"],
    "kube-controller-manager": ["kube-controller-manager", "controller-manager", "sig/apps"],
}

# canonical error signature -> regex. Order matters only for readability.
ERROR_SIGNATURES: dict[str, str] = {
    "OOMKilled": r"\bOOMKill(ed)?\b|out of memory|oom-kill",
    "CrashLoopBackOff": r"CrashLoopBackOff|back-off restarting failed container",
    "ImagePullBackOff": r"ImagePullBackOff|ErrImagePull",
    "context deadline exceeded": r"context deadline exceeded",
    "mvcc: database space exceeded": r"database space exceeded",
    "i/o timeout": r"i/o timeout",
    "PLEG is not healthy": r"PLEG is not healthy",
    "x509 certificate expired": r"x509: certificate has expired",
    "TLS verify failed": r"tls: failed to verify certificate",
    "leader changed": r"leader changed|leader election lost",
    "failed to create shim task": r"failed to create shim task",
    "FailedScheduling": r"FailedScheduling|nodes are available",
    "DiskPressure": r"DiskPressure|low on resource: ephemeral-storage",
    "502 Bad Gateway": r"\b502\b|bad gateway",
    "DNS loop detected": r"Loop \(.*\) detected",
    "conntrack drop": r"conntrack",
}

_SIG_RE = {name: re.compile(rx, re.IGNORECASE) for name, rx in ERROR_SIGNATURES.items()}
_COMP_RE = {
    name: re.compile(r"(?<![\w-])(" + "|".join(re.escape(a) for a in aliases) + r")(?![\w-])",
                     re.IGNORECASE)
    for name, aliases in COMPONENTS.items()
}


def find_error_signatures(text: str) -> list[str]:
    return [name for name, rx in _SIG_RE.items() if rx.search(text)]


def first_error_signature(text: str) -> str | None:
    found = find_error_signatures(text)
    return found[0] if found else None


def find_components(text: str) -> list[str]:
    return [name for name, rx in _COMP_RE.items() if rx.search(text)]


def canonical_component(name: str) -> str | None:
    """Map a free-text component name ("apiserver", "dns") to its canonical form."""
    name = name.strip().lower()
    for canon, aliases in COMPONENTS.items():
        if name == canon or name in (a.lower() for a in aliases):
            return canon
    return None
