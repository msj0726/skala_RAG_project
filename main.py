import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from agents import Synthesis, evaluate, refs_in, validate
from retrieval import ROOT, VectorRetriever
from report import render


class State(TypedDict, total=False):
    mode: str
    as_of: str
    market: list
    domain: list
    market_trace: list
    domain_trace: list
    synthesis: dict
    sources: list
    corpus_pages: int
    embedding_model: str
    outputs: dict


def build_graph(retriever, snapshot, output):
    def perspective(name):
        def run(state):
            results, traces = [], []
            for tech in ("MLA", "PNM"):
                print(f"[{name}] {tech} / {state['mode']}", flush=True)
                checkpoint = output / "checkpoints" / f"{name}_{tech}.json"
                cache_key = {"index": retriever.fingerprint,
                             "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini-2025-04-14"),
                             "rubric_version": "2026-09-23-throughput-v7"}
                cached = json.loads(checkpoint.read_text()) if state["mode"] == "live" and checkpoint.exists() else None
                if cached and cached.get("key") == cache_key:
                    result, trace = cached["result"], cached["trace"]
                    if "adjudication" not in trace:
                        reviewed = next(item for item in snapshot[name] if item["tech"] == tech)
                        source_set = [r["source"] for r in retriever.records if r["source"]["tech"] == tech]
                        chunks = [c for round_ in trace["rounds"] for c in round_["chunks"]]
                        proposal = result
                        result = validate(reviewed, name, source_set, chunks)
                        trace["adjudication"] = {"policy": "reviewed evidence controls report; live proposal retained for audit",
                                                  "proposal": proposal,
                                                  "matched_reviewed_evidence": proposal == result}
                        checkpoint.write_text(json.dumps({"key": cache_key, "result": result, "trace": trace}, ensure_ascii=False))
                    print(f"[{name}] {tech} verified checkpoint", flush=True)
                else:
                    result, trace = evaluate(tech, name, retriever, state["mode"], snapshot)
                    if state["mode"] == "live":
                        checkpoint.parent.mkdir(parents=True, exist_ok=True)
                        temporary = checkpoint.with_suffix(".pending")
                        temporary.write_text(json.dumps({"key": cache_key, "result": result, "trace": trace}, ensure_ascii=False))
                        temporary.replace(checkpoint)
                results.append(result)
                traces.append(trace)
            return {name: results, name + "_trace": traces}
        return run

    def synthesize(state):
        value = Synthesis.model_validate(snapshot["synthesis"]).model_dump()
        allowed = set(refs_in({"market": state["market"], "domain": state["domain"]}))
        if set(value["refs"]) - allowed:
            raise ValueError("Synthesis introduced an unevaluated reference")
        if len(value["summary"]) > 650:
            raise ValueError("SUMMARY exceeds the half-page budget")
        return {"synthesis": value}

    def write_report(state):
        return {"outputs": render(state, output)}

    graph = StateGraph(State)
    graph.add_node("market_agent", perspective("market"))
    graph.add_node("domain_agent", perspective("domain"))
    graph.add_node("synthesis_agent", synthesize)
    graph.add_node("report_writer", write_report)
    graph.add_edge(START, "market_agent")
    graph.add_edge(START, "domain_agent")
    graph.add_edge(["market_agent", "domain_agent"], "synthesis_agent")
    graph.add_edge("synthesis_agent", "report_writer")
    graph.add_edge("report_writer", END)
    return graph.compile()


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="MLA / CXL-PNM multi-agent evaluation")
    parser.add_argument("--mode", choices=["live", "replay"], default="live")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.mode == "live" and not os.getenv("OPENAI_API_KEY", "").strip():
        parser.error(".env에 OPENAI_API_KEY를 입력하세요. 키 없는 검증은 --mode replay입니다.")
    output = (args.output or ROOT / "output" / args.mode).resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads((ROOT / "data/reviewed_evidence.json").read_text())
    retriever = VectorRetriever()
    graph = build_graph(retriever, snapshot, output)
    (output / "graph.mmd").write_text(graph.get_graph().draw_mermaid())
    state = graph.invoke({"mode": args.mode, "as_of": "2026-09-23",
                          "sources": [r["source"] for r in retriever.records],
                          "corpus_pages": sum(len(r["pages"]) for r in retriever.records),
                          "embedding_model": retriever.model_name})
    web_evidence = {}
    for perspective in ("market", "domain"):
        for trace in state[perspective + "_trace"]:
            proposal_refs = set(refs_in(trace.get("adjudication", {}).get("proposal", {})))
            for round_ in trace["rounds"]:
                for evidence in round_.get("verified_web_evidence", []):
                    source_id = evidence["source"]["id"]
                    web_evidence[source_id] = dict(evidence, perspective=perspective,
                                                   tech=trace["tech"], proposed_citation=source_id in proposal_refs,
                                                   final_citation=False,
                                                   verification="HTTPS approved publisher; page fetched; relevant original text and SHA-256 recorded; claim not human-reviewed")
    state["web_evidence"] = list(web_evidence.values())
    (output / "web_evidence.json").write_text(json.dumps(state["web_evidence"], ensure_ascii=False, indent=2))
    state["execution"] = {"completed_at": datetime.now(timezone.utc).isoformat(),
                          "llm_model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini-2025-04-14") if args.mode == "live" else None,
                          "index_fingerprint": retriever.fingerprint,
                          "documents": [{k: v for k, v in r.items() if k != "pages"} for r in retriever.records]}
    (output / "evaluation.json").write_text(json.dumps(state, ensure_ascii=False, indent=2))
    print(json.dumps(state["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
