"""CLI: python -m ops_assistant.ingest [--issues FILE] [--logs DIR] [--reset]

With no arguments it ingests the bundled sample data from DATA_DIR.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

from ..config import get_settings
from ..embeddings import get_embedder
from ..store import PgStore
from .pipeline import ingest


def main() -> None:
    settings = get_settings()
    data_dir = Path(settings.data_dir)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issues", type=Path, default=data_dir / "issues.json")
    parser.add_argument("--logs", type=Path, default=data_dir / "logs")
    parser.add_argument("--reset", action="store_true",
                        help="drop and recreate tables (needed after changing EMBEDDING_DIM)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    store = PgStore(settings.database_url, settings.embedding_dim)
    try:
        report = ingest(
            store, get_embedder(settings), settings,
            args.issues if args.issues.exists() else None,
            args.logs if args.logs.exists() else None,
            reset=args.reset,
        )
    finally:
        store.close()
    print("ingestion complete:", asdict(report))


if __name__ == "__main__":
    main()
