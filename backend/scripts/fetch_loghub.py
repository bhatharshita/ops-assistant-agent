"""Download a public Loghub sample log (2,000 lines) for ingestion.

Loghub (https://github.com/logpai/loghub) is a public collection of system logs
for research. Each dataset has a small `<Name>_2k.log` sample in the repository.

    python scripts/fetch_loghub.py --dataset OpenStack
    python -m ops_assistant.ingest --logs ../data/raw/loghub
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

BASE = "https://raw.githubusercontent.com/logpai/loghub/master/{name}/{name}_2k.log"
DATASETS = ["OpenStack", "HDFS", "Hadoop", "Spark", "Zookeeper", "Linux", "Apache", "OpenSSH"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="OpenStack", choices=DATASETS)
    p.add_argument("--out-dir", type=Path, default=Path("../data/raw/loghub"))
    args = p.parse_args()

    url = BASE.format(name=args.dataset)
    resp = httpx.get(url, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.dataset.lower()}.log"
    out.write_text(resp.text)
    print(f"wrote {len(resp.text.splitlines())} lines from {url} to {out}")


if __name__ == "__main__":
    main()
