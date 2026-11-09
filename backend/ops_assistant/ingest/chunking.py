"""Chunking strategies.

Issues: split on paragraphs, pack paragraphs up to `size` characters, carry an
overlap tail into the next chunk, and prefix every chunk with the issue header so
a chunk taken from the middle of a thread still says which issue it belongs to.

Logs: two kinds of chunks per file
  1. digests  - one chunk per (component, error signature) with counts, time range,
                top nodes and an example line. Good for "what errors does X show?".
  2. windows  - consecutive raw lines, so specific messages stay searchable.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from ..store import Chunk, LogLine


def split_text(text: str, size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:  # hard-split paragraphs that are longer than a chunk
        while len(p) > size:
            cut = p.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            pieces.append(p[:cut])
            p = p[cut:].strip()
        pieces.append(p)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail  # start on a word
            current = f"{tail}\n\n{piece}" if tail else piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def chunk_issue(doc_id: str, text: str, size: int, overlap: int) -> list[Chunk]:
    header, _, rest = text.partition("\n\n")
    bodies = split_text(rest, max(size - len(header), 200), overlap) or [""]
    return [
        Chunk(id=f"{doc_id}#{i}", document_id=doc_id, source_type="issue",
              content=f"{header}\n\n{body}".strip())
        for i, body in enumerate(bodies)
    ]


def chunk_logs(doc_id: str, lines: list[LogLine], window: int = 25) -> list[Chunk]:
    chunks: list[Chunk] = []

    groups: dict[tuple[str, str], list[LogLine]] = defaultdict(list)
    for line in lines:
        if line.error_signature:
            groups[(line.component or "unknown", line.error_signature)].append(line)
    for n, ((component, sig), group) in enumerate(sorted(groups.items())):
        levels = Counter(x.level for x in group).most_common(1)[0][0]
        nodes = Counter(x.node for x in group if x.node).most_common(3)
        times = sorted(x.ts for x in group if x.ts)
        span = f"between {times[0]} and {times[-1]}" if times else ""
        node_txt = ", ".join(f"{node} ({c})" for node, c in nodes) or "n/a"
        content = (
            f"Log digest for {component}: {len(group)} {levels} lines with error '{sig}' {span}.\n"
            f"Most affected nodes: {node_txt}.\n"
            f"Example: {group[-1].message}"
        )
        chunks.append(Chunk(id=f"{doc_id}#digest{n}", document_id=doc_id, source_type="log",
                            content=content))

    for start in range(0, len(lines), window):
        block = lines[start:start + window]
        content = "\n".join(
            " ".join(x for x in (ln.ts, ln.level, ln.component, ln.node, ln.message) if x)
            for ln in block
        )
        chunks.append(Chunk(id=f"{doc_id}#w{start // window}", document_id=doc_id,
                            source_type="log", content=content))
    return chunks
