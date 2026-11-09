"""Turn raw files into Documents, text and structured log lines.

Issues file format (written by scripts/fetch_github_issues.py, same as the sample):
    {"repo": "owner/name", "issues": [{"number", "title", "state", "labels": [str],
      "created_at", "body", "comments": [str], "url"?}]}

Log line format of the sample logs:
    2026-10-14T03:12:45.123Z ERROR kubelet node-4 <message>
Lines that don't match (e.g. Loghub datasets) still load: level is guessed and the
component defaults to the file name.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..store import Document, LogLine
from .signatures import canonical_component, first_error_signature

LOG_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\S+)\s+(?P<level>DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\s+"
    r"(?P<component>\S+)\s+(?P<node>\S+)\s+(?P<message>.*)$"
)
LEVEL_RE = re.compile(r"\b(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b")


@dataclass
class IssueRecord:
    document: Document
    text: str  # full text: header + body + comments
    labels: list[str]


def load_issues(path: Path) -> list[IssueRecord]:
    data = json.loads(path.read_text())
    repo = data.get("repo", "unknown")
    records = []
    for raw in data["issues"]:
        labels = list(raw.get("labels", []))
        header = (f"Issue #{raw['number']}: {raw['title']}\n"
                  f"State: {raw.get('state', 'unknown')} | Labels: {', '.join(labels) or 'none'}")
        parts = [header, (raw.get("body") or "").strip()]
        parts += [f"Comment {i}: {c.strip()}" for i, c in enumerate(raw.get("comments", []), 1)]
        doc = Document(
            id=f"issue:{raw['number']}",
            source_type="issue",
            title=f"#{raw['number']} {raw['title']}",
            url=raw.get("url"),
            metadata={"repo": repo, "number": raw["number"], "state": raw.get("state"),
                      "labels": labels},
            created_at=raw.get("created_at"),
        )
        records.append(IssueRecord(doc, "\n\n".join(p for p in parts if p), labels))
    return records


def parse_log_line(line: str, document_id: str, default_component: str) -> LogLine | None:
    line = line.rstrip()
    if not line:
        return None
    m = LOG_RE.match(line)
    if m:
        level = m["level"].replace("WARNING", "WARN")
        component = canonical_component(m["component"]) or m["component"]
        return LogLine(document_id, m["ts"], level, component, m["node"], m["message"],
                       first_error_signature(m["message"]))
    lm = LEVEL_RE.search(line)
    level = lm.group(1).replace("WARNING", "WARN") if lm else None
    return LogLine(document_id, None, level, default_component, None, line,
                   first_error_signature(line))


def load_log_file(path: Path) -> tuple[Document, list[LogLine]]:
    doc_id = f"log:{path.name}"
    component = canonical_component(path.stem) or path.stem
    lines = [
        parsed
        for raw in path.read_text(errors="replace").splitlines()
        if (parsed := parse_log_line(raw, doc_id, component)) is not None
    ]
    doc = Document(id=doc_id, source_type="log", title=f"{path.name} logs",
                   metadata={"component": component, "lines": len(lines)})
    return doc, lines
