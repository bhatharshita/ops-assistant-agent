"""Deterministic, rule-based stand-ins for the LLM calls.

Used when no API key is configured (local demo, CI, tests). They follow the same
contracts as the LLM versions - the router returns a route, the analyst returns an
AIMessage with tool_calls, the writer returns text with [n] citations - so the
graph, tools and API are exercised identically with or without an LLM.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage

from ..ingest.signatures import find_components, find_error_signatures

LOG_WORDS = re.compile(
    r"\b(logs?|logged|how many|count|most|least|which node|nodes?|when|spike|top|"
    r"frequent|occurr\w*|lines?|errors? (are|were) we)\b", re.I)
ISSUE_WORDS = re.compile(
    r"\b(issues?|fix(ed|es)?|workaround|resolv\w*|solution|why|cause[sd]?|how (do|can|to|was|"
    r"should)|known|bug|recommend\w*|prevent)\b", re.I)
GRAPH_WORDS = re.compile(
    r"\b(related|relationships?|linked|connected|associated|same component|which components|"
    r"what else|neighbou?rs?|graph)\b", re.I)
DOMAIN_WORDS = re.compile(
    r"\b(pods?|nodes?|cluster|containers?|k8s|kubernetes|logs?|issues?|errors?|crash\w*|dns|"
    r"certificates?|deployments?|images?|memory|disk|etcd|scheduler|ingress|network|latency|"
    r"timeouts?|restarts?|evict\w*|upgrade)\b", re.I)
ISSUE_NUMBER = re.compile(r"#(\d+)|issue[: ](\d+)", re.I)


def route_question(question: str) -> tuple[str, str]:
    q = question
    in_domain = bool(DOMAIN_WORDS.search(q) or find_components(q) or find_error_signatures(q))
    if not in_domain:
        return "out_of_scope", "Explain what the assistant can help with."
    if GRAPH_WORDS.search(q):
        return "graph", "Look up the entity in the knowledge graph, then summarise its links."
    logs, issues = bool(LOG_WORDS.search(q)), bool(ISSUE_WORDS.search(q))
    if logs and issues:
        return "mixed", "Count the relevant log lines, then find issues explaining the fix."
    if logs:
        return "logs", "Aggregate the structured log lines and report exact counts."
    return "issues", "Search the GitHub issues and summarise causes and fixes."


def _group_by(q: str, component: str | None) -> str:
    ql = q.lower()
    if re.search(r"\bnodes?\b", ql):
        return "node"
    if re.search(r"\b(which|what) (errors?|kinds?|types?)\b|error types", ql) or (
            component and "component" not in ql):
        return "error_signature"
    if "level" in ql:
        return "level"
    return "component"


def plan_tool_calls(question: str, route: str) -> list[dict[str, Any]]:
    """Pick tools and arguments the way an LLM analyst would for this route."""
    components = find_components(question)
    signatures = find_error_signatures(question)
    component = components[0] if components else None
    calls: list[dict[str, Any]] = []

    if route in ("logs", "mixed"):
        args: dict[str, Any] = {"group_by": _group_by(question, component)}
        if component and args["group_by"] != "component":
            args["component"] = component
        if signatures and args["group_by"] != "error_signature":
            args["error_signature"] = signatures[0]
        if re.search(r"\berrors?\b", question, re.I) and not signatures:
            args["level"] = "ERROR"
        calls.append({"name": "log_stats", "args": args})

    if route == "graph":
        m = ISSUE_NUMBER.search(question)
        entity = (f"issue:{m.group(1) or m.group(2)}" if m
                  else (signatures[0] if signatures else component))
        if entity:
            calls.append({"name": "graph_neighbors", "args": {"entity": entity}})

    if route == "issues":
        m = ISSUE_NUMBER.search(question)
        if m:
            calls.append({"name": "get_issue", "args": {"number": int(m.group(1) or m.group(2))}})

    return calls


def analyst_message(question: str, route: str, already_called: bool) -> AIMessage:
    if already_called:
        return AIMessage(content="done")
    calls = plan_tool_calls(question, route)
    return AIMessage(
        content="" if calls else "done",
        tool_calls=[{"name": c["name"], "args": c["args"], "id": f"call_{uuid.uuid4().hex[:8]}",
                     "type": "tool_call"} for c in calls],
    )


def _resolution(content: str) -> str | None:
    """Pull the most useful sentence(s) out of an issue chunk."""
    m = re.search(r"Resolution:\s*(.+?)(?:\n|$)", content)
    if m:
        return m.group(1).strip()
    comments = re.findall(r"Comment \d+:\s*(.+?)(?:\n|$)", content)
    return comments[-1].strip() if comments else None


def compose_answer(question: str, citations: list[dict[str, Any]],
                   tool_results: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for tr in tool_results:
        result = tr["result"]
        if tr["name"] == "log_stats" and result.get("rows"):
            rows = result["rows"][:5]
            items = "; ".join(f"{r['key']}: {r['count']} lines" for r in rows)
            top = rows[0]
            parts.append(
                f"From the logs (grouped by {result['group_by']}): {items}. "
                f"The top one is {top['key']} with {top['count']} lines between "
                f"{top['first_seen']} and {top['last_seen']}; latest example: \"{top['example']}\"."
            )
        elif tr["name"] == "graph_neighbors" and result.get("found"):
            links = ", ".join(f"{n['label']} ({n['relation']}, weight {n['weight']})"
                              for n in result["neighbors"][:8])
            issues = ", ".join(i["label"] for i in result.get("related_issues", [])[:6])
            parts.append(f"In the knowledge graph, {result['node']['label']} is linked to: "
                         f"{links}." + (f" Related issues: {issues}." if issues else ""))
        elif tr["name"] == "get_issue" and "title" in result:
            fix = _resolution(result["content"])
            parts.append(f"Issue {result['title']}" + (f": {fix}" if fix else "."))

    issue_lines = []
    for n, src in enumerate(citations, start=1):
        if src["source_type"] == "issue" and len(issue_lines) < 3:
            fix = _resolution(src["content"])
            if fix:
                issue_lines.append(f"{src['title']}: {fix} [{n}]")
        elif src["source_type"] == "log" and src["chunk_id"].split("#")[-1].startswith("digest") \
                and not any(tr["name"] == "log_stats" for tr in tool_results):
            issue_lines.append(f"{src['content'].splitlines()[0]} [{n}]")
    if issue_lines:
        parts.append("Relevant evidence:\n- " + "\n- ".join(issue_lines))

    if not parts:
        return ("I could not find anything in the ingested issues or logs that answers this. "
                "Try naming a component (e.g. kubelet) or an error (e.g. OOMKilled).")
    return "\n\n".join(parts)
