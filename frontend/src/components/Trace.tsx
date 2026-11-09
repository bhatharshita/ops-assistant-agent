export type TraceStep = {
  kind: "route" | "retrieve" | "tool" | "result";
  label: string;
  detail?: string;
};

// Shows what the LangGraph nodes did, in order: route -> retrieve -> tool calls -> results.
export function Trace({ steps }: { steps: TraceStep[] }) {
  if (steps.length === 0) return <p className="muted">No steps yet.</p>;
  return (
    <ol className="trace">
      {steps.map((s, i) => (
        <li key={i} className={`step step-${s.kind}`}>
          <div className="step-label">{s.label}</div>
          {s.detail && (s.kind === "result"
            ? <details><summary>show</summary><pre>{s.detail}</pre></details>
            : <code>{s.detail}</code>)}
        </li>
      ))}
    </ol>
  );
}
