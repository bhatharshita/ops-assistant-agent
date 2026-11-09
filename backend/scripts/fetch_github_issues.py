"""Download public GitHub issues (with comments) into the format the ingester reads.

    python scripts/fetch_github_issues.py --repo kubernetes/kubernetes --label kind/bug \\
        --limit 200 --out ../data/raw/kubernetes_issues.json
    python -m ops_assistant.ingest --issues ../data/raw/kubernetes_issues.json \\
        --logs ../data/sample/logs

Works without a token (60 requests/hour); set GITHUB_TOKEN for 5,000/hour.
Pull requests are skipped (the issues API returns both).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx

API = "https://api.github.com"


def get(client: httpx.Client, url: str, **params) -> httpx.Response:
    resp = client.get(url, params=params)
    if resp.status_code == 403 and resp.headers.get("x-ratelimit-remaining") == "0":
        wait = int(resp.headers.get("x-ratelimit-reset", time.time() + 60)) - int(time.time())
        raise SystemExit(f"GitHub rate limit hit; retry in {max(wait, 0)}s or set GITHUB_TOKEN")
    resp.raise_for_status()
    return resp


def fetch(repo: str, label: str | None, state: str, limit: int, max_comments: int) -> list[dict]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "ops-assistant-ingest"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    issues: list[dict] = []
    with httpx.Client(headers=headers, timeout=30) as client:
        page = 1
        while len(issues) < limit:
            params = {"state": state, "per_page": 100, "page": page}
            if label:
                params["labels"] = label
            batch = get(client, f"{API}/repos/{repo}/issues", **params).json()
            if not batch:
                break
            for raw in batch:
                if "pull_request" in raw:
                    continue
                comments = []
                if max_comments and raw.get("comments"):
                    comments = [c["body"] or "" for c in get(
                        client, raw["comments_url"], per_page=max_comments).json()]
                issues.append({
                    "number": raw["number"], "title": raw["title"], "state": raw["state"],
                    "labels": [lb["name"] for lb in raw.get("labels", [])],
                    "created_at": raw["created_at"], "url": raw["html_url"],
                    "body": (raw.get("body") or "")[:20_000], "comments": comments,
                })
                if len(issues) >= limit:
                    break
            page += 1
    return issues


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="kubernetes/kubernetes")
    p.add_argument("--label", default="kind/bug")
    p.add_argument("--state", default="all", choices=["open", "closed", "all"])
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--max-comments", type=int, default=5,
                   help="comments per issue (each costs one API call; 0 to skip)")
    p.add_argument("--out", type=Path, default=Path("../data/raw/github_issues.json"))
    args = p.parse_args()

    issues = fetch(args.repo, args.label, args.state, args.limit, args.max_comments)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"repo": args.repo, "issues": issues}, indent=1))
    print(f"wrote {len(issues)} issues from {args.repo} to {args.out}")


if __name__ == "__main__":
    main()
