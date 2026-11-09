// Types mirror the events emitted by the backend (ops_assistant/agent/graph.py).

export type Source = {
  chunk_id: string;
  document_id: string;
  source_type: "issue" | "log";
  title: string;
  url: string | null;
  content: string;
  score: number;
};

export type AgentEvent =
  | { type: "route"; route: string; plan: string }
  | { type: "sources"; sources: Source[] }
  | { type: "tool_call"; name: string; args: Record<string, unknown> }
  | { type: "tool_result"; name: string; result: unknown }
  | { type: "token"; text: string }
  | { type: "done"; answer: string; route: string; citations: Source[]; latency_ms: number }
  | { type: "error"; message: string };

export type Health = {
  status: string;
  database: boolean;
  llm_provider: string;
  embedding_provider: string;
  retrieval_mode: string;
  index: Record<string, number>;
};

const API = "/api";

export async function getHealth(): Promise<Health> {
  const res = await fetch(`${API}/health`);
  if (!res.ok) throw new Error(`health check failed: ${res.status}`);
  return res.json();
}

/**
 * Parse server-sent events out of a growing text buffer.
 * Returns the complete events and the unconsumed remainder (a partial event).
 */
export function parseSSE(buffer: string): { events: AgentEvent[]; rest: string } {
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: AgentEvent[] = [];
  for (const block of blocks) {
    const data = block
      .split("\n")
      .filter((line) => line.startsWith("data: "))
      .map((line) => line.slice(6))
      .join("\n");
    if (data) events.push(JSON.parse(data) as AgentEvent);
  }
  return { events, rest };
}

/**
 * POST the question and call onEvent for each streamed event.
 * EventSource only supports GET, so we read the fetch body stream ourselves.
 */
export async function streamChat(
  question: string,
  onEvent: (e: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(`request failed: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parsed = parseSSE(buffer);
    buffer = parsed.rest;
    parsed.events.forEach(onEvent);
  }
}
