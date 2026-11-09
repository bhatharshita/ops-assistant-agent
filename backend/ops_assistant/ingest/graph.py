"""Build a small knowledge graph linking issues, components and errors.

    (issue)-[:AFFECTS]->(component)     from labels + text
    (issue)-[:MENTIONS]->(error)        from text, via error-signature regexes
    (error)-[:SEEN_IN]->(component)     from log lines, weight = number of lines

The graph lets the agent answer relationship questions that plain vector search
handles poorly, e.g. "which other issues involve the same component as #1006?"
"""

from __future__ import annotations

from collections import Counter

from ..store import LogLine
from .loaders import IssueRecord
from .signatures import find_components, find_error_signatures

Nodes = dict[str, tuple[str, str]]  # id -> (kind, label)
Edges = dict[tuple[str, str, str], int]  # (src, dst, kind) -> weight


def component_id(name: str) -> str:
    return f"component:{name}"


def error_id(name: str) -> str:
    return f"error:{name}"


def build_graph(issues: list[IssueRecord], log_lines: list[LogLine]) -> tuple[Nodes, Edges]:
    nodes: Nodes = {}
    edges: Edges = {}

    for rec in issues:
        iid = rec.document.id
        nodes[iid] = ("issue", rec.document.title)
        components = set(find_components(" ".join(rec.labels))) | set(find_components(rec.text))
        for comp in components:
            nodes[component_id(comp)] = ("component", comp)
            edges[(iid, component_id(comp), "AFFECTS")] = 1
        for sig in find_error_signatures(rec.text):
            nodes[error_id(sig)] = ("error", sig)
            edges[(iid, error_id(sig), "MENTIONS")] = 1

    seen = Counter((ln.error_signature, ln.component) for ln in log_lines
                   if ln.error_signature and ln.component)
    for (sig, comp), count in seen.items():
        nodes[error_id(sig)] = ("error", sig)
        nodes[component_id(comp)] = ("component", comp)
        edges[(error_id(sig), component_id(comp), "SEEN_IN")] = count

    return nodes, edges
