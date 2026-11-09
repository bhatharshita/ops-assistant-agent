import type { Source } from "../api";

export function SourcesPanel({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return <p className="muted">No sources for this answer.</p>;
  return (
    <ol className="sources">
      {sources.map((s, i) => (
        <li key={`${s.chunk_id}-${i}`}>
          <div className="source-head">
            <span className="num">[{i + 1}]</span>
            <span className={`badge badge-${s.source_type}`}>{s.source_type}</span>
            {s.url ? <a href={s.url} target="_blank" rel="noreferrer">{s.title}</a> : <span>{s.title}</span>}
          </div>
          <pre>{s.content.length > 600 ? `${s.content.slice(0, 600)}…` : s.content}</pre>
          <div className="meta">{s.chunk_id} · score {s.score.toFixed(4)}</div>
        </li>
      ))}
    </ol>
  );
}
