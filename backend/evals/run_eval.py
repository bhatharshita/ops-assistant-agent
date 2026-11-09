"""Evaluation harness.

    python -m evals.run_eval                       # current settings (hybrid by default)
    python -m evals.run_eval --retrieval dense     # baseline for the before/after table
    python -m evals.run_eval --no-judge            # skip LLM-as-judge even if a key is set

Metrics
- route_accuracy   router picked the labelled route
- hit@k / MRR      a labelled relevant document appears in the retriever's top-k
                   (document level; MRR = mean of 1 / rank of the first relevant hit)
- keyword_recall   share of `must_include` facts present in the final answer
                   (cheap, deterministic proxy for correctness)
- judge_score      LLM-as-judge 1-5 for correctness vs the reference answer and
                   faithfulness to the cited sources (only when an LLM is configured)
- latency p50/p95  wall-clock time of a full agent run
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ops_assistant.config import get_settings
from ops_assistant.service import OpsAssistant

HERE = Path(__file__).parent

JUDGE_PROMPT = """You grade answers from an operations assistant.
Given the question, a reference answer, the assistant's answer and the sources it cited,
score from 1 (wrong) to 5 (fully correct):
- correctness: does the answer agree with the reference on the key facts and numbers?
- faithfulness: is every claim supported by the cited sources / tool results?"""


class JudgeScore(BaseModel):
    correctness: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    reason: str


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def retrieval_rank(doc_ids: list[str], relevant: list[str]) -> int | None:
    for rank, doc_id in enumerate(doc_ids, start=1):
        if doc_id in relevant:
            return rank
    return None


def keyword_recall(answer: str, must_include: list[str]) -> float:
    if not must_include:
        return 1.0
    text = answer.lower()
    return sum(kw.lower() in text for kw in must_include) / len(must_include)


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def judge(assistant: OpsAssistant, item: dict[str, Any], result: dict[str, Any]) -> JudgeScore:
    cited = "\n".join(f"[{n}] {c['title']}: {c['content'][:500]}"
                      for n, c in enumerate(result["citations"], 1))
    tools = json.dumps([{"tool": t["name"], "result": t["result"]}
                        for t in result["tool_results"]], default=str)[:3000]
    msg = (f"Question: {item['question']}\nReference: {item['reference']}\n"
           f"Answer: {result['answer']}\nSources:\n{cited}\nTool results: {tools}")
    return assistant.deps.llm.with_structured_output(JudgeScore).invoke(
        [SystemMessage(JUDGE_PROMPT), HumanMessage(msg)])


def run(retrieval: str | None, use_judge: bool, label: str | None) -> dict[str, Any]:
    settings = get_settings()
    if retrieval:
        settings = settings.model_copy(update={"retrieval_mode": retrieval})
    assistant = OpsAssistant(settings)
    use_judge = use_judge and assistant.deps.llm is not None
    k = settings.top_k
    rows = []
    try:
        for item in load_dataset(HERE / "dataset.jsonl"):
            # retrieval is scored on the raw retriever, so it isolates the RAG step
            hits = assistant.retriever.search(item["question"], k=k)
            rank = retrieval_rank([h.document_id for h in hits], item["relevant"])
            t0 = time.perf_counter()
            result = assistant.ask(item["question"])
            latency = (time.perf_counter() - t0) * 1000
            row = {
                "id": item["id"], "question": item["question"],
                "expected_route": item["route"], "route": result["route"],
                "rank": rank, "keyword_recall": keyword_recall(result["answer"],
                                                               item["must_include"]),
                "latency_ms": round(latency, 1), "answer": result["answer"],
            }
            if use_judge:
                score = judge(assistant, item, result)
                row.update(correctness=score.correctness, faithfulness=score.faithfulness,
                           judge_reason=score.reason)
            rows.append(row)
            print(f"{item['id']} route={row['route']:<12} rank={rank!s:<4} "
                  f"recall={row['keyword_recall']:.2f} {row['latency_ms']:>7.1f}ms")
    finally:
        assistant.close()

    retrieval_rows = [r for r in rows if r["expected_route"] != "out_of_scope"]
    latencies = [r["latency_ms"] for r in rows]
    summary = {
        "label": label or f"{settings.retrieval_mode}-{assistant.mode}",
        "llm_provider": assistant.mode,
        "embedding_provider": settings.resolved_embedding_provider(),
        "retrieval_mode": settings.retrieval_mode,
        "n": len(rows),
        "route_accuracy": round(statistics.mean(r["route"] == r["expected_route"]
                                                for r in rows), 3),
        f"hit@{k}": round(statistics.mean(r["rank"] is not None for r in retrieval_rows), 3),
        "mrr": round(statistics.mean(1 / r["rank"] if r["rank"] else 0
                                     for r in retrieval_rows), 3),
        "keyword_recall": round(statistics.mean(r["keyword_recall"] for r in rows), 3),
        "latency_p50_ms": round(percentile(latencies, 50), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
    }
    if use_judge:
        summary["judge_correctness"] = round(statistics.mean(r["correctness"] for r in rows), 2)
        summary["judge_faithfulness"] = round(statistics.mean(r["faithfulness"] for r in rows),
                                              2)
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval", choices=["dense", "hybrid"])
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--out", type=Path, help="write full results JSON here")
    args = parser.parse_args()

    report = run(args.retrieval, not args.no_judge, args.label)
    print("\nSummary")
    for key, value in report["summary"].items():
        print(f"  {key:<20} {value}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
