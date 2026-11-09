// Renders answer text, keeping line breaks and highlighting [n] citations.
export function Answer({ text, pending }: { text: string; pending: boolean }) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <div className="answer-text">
      {parts.map((p, i) =>
        /^\[\d+\]$/.test(p) ? <sup key={i} className="cite">{p}</sup> : <span key={i}>{p}</span>,
      )}
      {pending && <span className="cursor">▍</span>}
    </div>
  );
}
