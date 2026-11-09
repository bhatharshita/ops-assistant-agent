import { useEffect, useRef, useState } from "react";
import { type AgentEvent, getHealth, type Health, type Source, streamChat } from "./api";
import { Answer } from "./components/Answer";
import { SourcesPanel } from "./components/SourcesPanel";
import { Trace, type TraceStep } from "./components/Trace";

type Turn = {
  question: string;
  answer: string;
  route?: string;
  sources: Source[];
  trace: TraceStep[];
  latencyMs?: number;
  error?: string;
  done: boolean;
};

const EXAMPLES = [
  "Which node had the most OOMKilled containers?",
  "How was the etcd database space exceeded problem fixed?",
  "Which issues are related to etcd?",
  "What errors does the kubelet log most often and how do I fix them?",
];

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);
  const [tab, setTab] = useState<"sources" | "trace">("sources");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    // block body on purpose: an effect must not return scrollIntoView's result
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  // Apply one streamed event to the turn being answered (always the last one).
  const apply = (e: AgentEvent) =>
    setTurns((prev) => {
      const turns = [...prev];
      const t = { ...turns[turns.length - 1] };
      switch (e.type) {
        case "route":
          t.route = e.route;
          t.trace = [...t.trace, { kind: "route", label: `route: ${e.route}`, detail: e.plan }];
          break;
        case "sources":
          t.sources = [...t.sources, ...e.sources];
          t.trace = [...t.trace, { kind: "retrieve", label: `retrieved ${e.sources.length} chunks` }];
          break;
        case "tool_call":
          t.trace = [...t.trace, { kind: "tool", label: `call ${e.name}`, detail: JSON.stringify(e.args) }];
          break;
        case "tool_result":
          t.trace = [...t.trace, { kind: "result", label: `${e.name} result`, detail: JSON.stringify(e.result, null, 2) }];
          break;
        case "token":
          t.answer += e.text;
          break;
        case "done":
          t.answer = e.answer;
          t.sources = e.citations; // numbered exactly as cited in the answer
          t.latencyMs = e.latency_ms;
          t.done = true;
          break;
        case "error":
          t.error = e.message;
          t.done = true;
          break;
      }
      turns[turns.length - 1] = t;
      return turns;
    });

  async function ask(question: string) {
    if (!question.trim() || busy) return;
    setInput("");
    setBusy(true);
    setTurns((prev) => [...prev, { question, answer: "", sources: [], trace: [], done: false }]);
    setSelected(turns.length);
    try {
      await streamChat(question, apply);
    } catch (err) {
      apply({ type: "error", message: String(err) });
    } finally {
      setBusy(false);
    }
  }

  const current = selected !== null ? turns[selected] : turns[turns.length - 1];

  return (
    <div className="app">
      <header>
        <h1>Ops Assistant</h1>
        <span className="health">
          {health
            ? `LLM: ${health.llm_provider} · embeddings: ${health.embedding_provider} · ${health.index.chunks ?? 0} chunks indexed`
            : "backend unreachable"}
        </span>
      </header>

      <main>
        <section className="chat">
          {turns.length === 0 && (
            <div className="empty">
              <p>Ask about the project's logs and GitHub issues.</p>
              <div className="examples">
                {EXAMPLES.map((q) => (
                  <button key={q} onClick={() => ask(q)}>{q}</button>
                ))}
              </div>
            </div>
          )}
          {turns.map((t, i) => (
            <article key={i} className={i === selected ? "turn selected" : "turn"} onClick={() => setSelected(i)}>
              <div className="question">{t.question}</div>
              <div className="answer">
                {t.route && <span className={`route route-${t.route}`}>{t.route}</span>}
                <Answer text={t.answer} pending={!t.done} />
                {t.error && <div className="error">Error: {t.error}</div>}
                {t.latencyMs !== undefined && <div className="meta">{t.latencyMs} ms · {t.sources.length} sources</div>}
              </div>
            </article>
          ))}
          <div ref={bottomRef} />
          <form onSubmit={(e) => { e.preventDefault(); ask(input); }}>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="e.g. Why are pods stuck in CrashLoopBackOff?"
              aria-label="Question"
            />
            <button type="submit" disabled={busy || !input.trim()}>{busy ? "Thinking…" : "Ask"}</button>
          </form>
        </section>

        <aside>
          <div className="tabs" role="tablist">
            <button role="tab" aria-selected={tab === "sources"} onClick={() => setTab("sources")}>Sources</button>
            <button role="tab" aria-selected={tab === "trace"} onClick={() => setTab("trace")}>Agent trace</button>
          </div>
          {current ? (
            tab === "sources" ? <SourcesPanel sources={current.sources} /> : <Trace steps={current.trace} />
          ) : (
            <p className="muted">Sources and the agent's steps appear here.</p>
          )}
        </aside>
      </main>
    </div>
  );
}
