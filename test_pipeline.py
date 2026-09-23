"""Run: python -m unittest -v. No network, model, or API required."""
import copy
import json
import unittest
from unittest.mock import patch, Mock

from agents import validate, ask, Plan, evaluate
from retrieval import ROOT, read_sources, fetch_source, verify_web_results
from scoring import growth, adoption, ecosystem, capacity, throughput


class RubricTests(unittest.TestCase):
    def test_attached_papers_are_the_rag_originals(self):
        for source in read_sources()[:2]:
            record = fetch_source(source)
            self.assertEqual(record["sha256"], source["sha256"])
            self.assertGreater(len(record["pages"]), 10)
            self.assertIn(source["title"].split(":")[0], record["pages"][0])

    def test_reviewed_web_sources_match_saved_originals(self):
        ledger = json.loads((ROOT / "data/source_verification.json").read_text())
        for item in ledger["sources"]:
            self.assertIn(item["id"], {s["id"] for s in read_sources()})
            self.assertEqual(len(item["sha256"]), 64)
            path = ROOT / "data/raw" / (item["id"] + ".json")
            if path.exists():
                record = json.loads(path.read_text())
                self.assertEqual(record["sha256"], item["sha256"])
                self.assertGreater(len(" ".join(record["pages"])), 1000)

    def test_web_evidence_requires_fetched_primary_page_and_topic(self):
        page = Mock(ok=True, content=b"DeepSeek MLA Multi-head Latent Attention framework implementation " * 5,
                    headers={"content-type": "text/html"})
        page.raise_for_status.return_value = None
        results = [{"url": "https://github.com/deepseek-ai/FlashMLA?utm_source=openai", "title": "FlashMLA"},
                   {"url": "https://example.org/claim", "title": "unsupported"}]
        with patch("retrieval.requests.get", return_value=page) as get:
            accepted, rejected = verify_web_results(results, "MLA")
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["source"]["url"], "https://github.com/deepseek-ai/FlashMLA")
        self.assertEqual(len(rejected), 1)
        self.assertFalse(get.call_args.kwargs["allow_redirects"])
        with patch("retrieval.requests.get", return_value=page):
            accepted, _ = verify_web_results(results[:1], "MLA", [{"id": "S10", "tech": "MLA", "url": "https://github.com/deepseek-ai/FlashMLA"}])
        self.assertEqual(accepted[0]["source"]["id"], "S10")

    def test_growth_and_ecosystem(self):
        self.assertEqual([growth(p)["label"] for p in ([0,0,0], [0,0,1], [0,1,1], [1,1,1], [2,1,1], [2,2,1], [2,2,2])],
                         ["낮음", "낮음", "보통", "보통", "높음", "높음", "매우 높음"])
        self.assertEqual([ecosystem([True]*n + [False]*(4-n), False)["label"] for n in range(5)], ["L0", "L2", "L2", "L3", "L3"])
        self.assertEqual(ecosystem([False]*4, True)["label"], "L1")
        for invalid in ([1, 2], [3, 0, 0], [True, 1, 1]):
            with self.assertRaises(ValueError):
                growth(invalid)

    def test_capacity_boundaries(self):
        self.assertEqual([capacity(expansion=x)["label"] for x in (1, 1.01, 2, 3.99, 4)], ["L0", "L1", "L2", "L2", "L3"])
        self.assertEqual(capacity(reduction_pct=93.3)["factor"], 14.925)
        self.assertEqual(capacity()["status"], "근거 없음")
        for invalid in (float("nan"), float("inf"), -1, True):
            with self.assertRaises(ValueError):
                capacity(expansion=invalid)

    def test_throughput_boundaries_and_reviewed_factors(self):
        self.assertEqual([throughput(x)["label"] for x in (None, 1, 1.01, 2, 3.99, 4)],
                         ["판정 유보", "L0", "L1", "L2", "L2", "L3"])
        for invalid in (float("nan"), float("inf"), -1, True):
            with self.assertRaises(ValueError):
                throughput(invalid)
        snapshot = json.loads((ROOT / "data/reviewed_evidence.json").read_text())
        self.assertEqual([throughput(x["throughput"]["improvement_factor"])["label"]
                          for x in snapshot["domain"]], ["L3", "L3"])

    def test_provenance_and_scope(self):
        snapshot = json.loads((ROOT / "data/reviewed_evidence.json").read_text())
        market = snapshot["market"][1]
        self.assertEqual(adoption(market["cases"]), "L3")
        self.assertEqual(adoption(market["cases"], "selected"), "L1")
        chunks = [{"source_id": s["id"]} for s in read_sources()]
        for perspective in ("market", "domain"):
            for item in snapshot[perspective]:
                validate(item, perspective, read_sources(), chunks)
        invalid = copy.deepcopy(market)
        invalid["growth"][0]["refs"] = ["invented"]
        with self.assertRaises(ValueError):
            validate(invalid, "market", read_sources(), chunks)
        reordered = copy.deepcopy(market)
        reordered["ecosystem"].reverse()
        self.assertEqual(validate(reordered, "market", read_sources(), chunks)["ecosystem"][0]["name"], "프레임워크")
        repeated = copy.deepcopy(market)
        repeated["ecosystem"][0] = repeated["ecosystem"][1]
        with self.assertRaises(ValueError):
            validate(repeated, "market", read_sources(), chunks)
        with self.assertRaises(ValueError):
            validate(market, "market", read_sources(), [])
        invalid = copy.deepcopy(snapshot["domain"][0])
        invalid["capacity"]["refs"] = ["S5"]
        with self.assertRaises(ValueError):
            validate(invalid, "domain", read_sources(), chunks)

    def test_openai_structured_contract_and_failure(self):
        reply = Mock(ok=True)
        reply.json.return_value = {"status":"completed", "output":[{"content":[{"type":"output_text", "text":'{"query":"MLA","missing_evidence":"cost"}'}]}]}
        with patch.dict("os.environ", {"OPENAI_API_KEY":"test-not-a-real-key"}), patch("agents.requests.post", return_value=reply) as post:
            self.assertEqual(ask(Plan, "plan", {})["query"], "MLA")
            self.assertFalse(post.call_args.kwargs["json"]["store"])
            self.assertTrue(post.call_args.kwargs["json"]["text"]["format"]["strict"])
            reply.json.return_value = {"status":"incomplete"}
            with self.assertRaises(RuntimeError):
                ask(Plan, "plan", {})

    def test_live_followup_is_bounded_and_scored(self):
        snapshot = json.loads((ROOT / "data/reviewed_evidence.json").read_text())
        retriever = Mock()
        retriever.records = [{"source": s, "pages": ["original"]} for s in read_sources()]
        retriever.search.side_effect = lambda q, tech, ids, k: [{"id": ids[0] + ":p1:t0", "source_id": ids[0], "text": "original"}]
        result = snapshot["market"][0]
        responses = [{"query":"first search", "missing_evidence":"cost"},
                     {"result":result, "needs_more":True, "followup_query":"specific followup"},
                     {"result":result, "needs_more":True, "followup_query":"must not cause third attempt"}]
        with patch("agents.ask", side_effect=responses), patch("agents.web_search", return_value=[{"url":"https://example.org"}]) as search:
            evaluated, trace = evaluate("MLA", "market", retriever, "live", snapshot)
        self.assertEqual(search.call_count, 2)
        self.assertIn("DeepSeek-V2 MLA KV cache", search.call_args.args[0])
        self.assertEqual(trace["rounds"][1]["query"], "specific followup")
        self.assertEqual(evaluated["scores"]["M1"]["score"], 4)

    def test_live_overclaim_does_not_enter_report(self):
        snapshot = json.loads((ROOT / "data/reviewed_evidence.json").read_text())
        proposal = copy.deepcopy(snapshot["market"][0])
        proposal["cases"][0]["level"] = 4  # Research misreported as commercial operation.
        retriever = Mock()
        retriever.records = [{"source": s, "pages": ["original"]} for s in read_sources()]
        retriever.search.side_effect = lambda q, tech, ids, k: [{"id":ids[0] + ":p1:t0", "source_id":ids[0], "text":"original"}]
        with patch("agents.ask", side_effect=[{"query":"search", "missing_evidence":""},
                                              {"result":proposal, "needs_more":False, "followup_query":""}]), \
             patch("agents.web_search", return_value=[{"url":"https://example.org"}]):
            assessed, trace = evaluate("MLA", "market", retriever, "live", snapshot)
        self.assertEqual(assessed["scores"]["M2_selected"], "L1")
        self.assertEqual(trace["adjudication"]["proposal"]["scores"]["M2_selected"], "L4")
        self.assertFalse(trace["adjudication"]["matched_reviewed_evidence"])

    def test_rejected_web_claim_stays_out_of_report(self):
        snapshot = json.loads((ROOT / "data/reviewed_evidence.json").read_text())
        proposal = copy.deepcopy(snapshot["market"][0])
        proposal["ecosystem"][0]["refs"] = ["Wtest"]
        retriever = Mock()
        retriever.records = [{"source": s, "pages": ["original"]} for s in read_sources()]
        retriever.search.side_effect = lambda q, tech, ids, k: [{"id": ids[0] + ":p1:t0", "source_id": ids[0], "text": "original"}]
        verified = [{"source": {"id": "Wtest", "tech": "MLA", "scope": "family", "url": "https://github.com/test"},
                     "pages": ["DeepSeek MLA"], "sha256": "test", "retrieved_at": "now"}]
        with patch("agents.ask", side_effect=[{"query": "search", "missing_evidence": ""},
                                              {"result": proposal, "needs_more": False, "followup_query": ""},
                                              {"result": proposal, "needs_more": False, "followup_query": ""}]), \
             patch("agents.web_search", return_value=[{"url": "https://github.com/test"}]), \
             patch("agents.verify_web_results", return_value=(verified, [])):
            assessed, trace = evaluate("MLA", "market", retriever, "live", snapshot)
        self.assertEqual(assessed["scores"]["M3"]["label"], "L2")
        self.assertIsNone(trace["adjudication"]["proposal"])
        self.assertIn("rejected", trace["correction"])

    def test_graph_joins_both_perspectives_before_report(self):
        from main import build_graph
        snapshot = json.loads((ROOT / "data/reviewed_evidence.json").read_text())
        retriever = Mock()
        retriever.records = [{"source": s, "pages": ["original"]} for s in read_sources()]
        retriever.search.side_effect = lambda q, tech, ids, k: [{"id":ids[0] + ":p1:t0", "source_id":ids[0], "text":"original"}]
        with patch("main.render", return_value={"pages":8}) as writer:
            result = build_graph(retriever, snapshot, ROOT / "tmp").invoke({"mode":"replay"})
        self.assertEqual(len(result["market"]), 2)
        self.assertEqual(len(result["domain"]), 2)
        writer.assert_called_once()
        self.assertIn("synthesis", writer.call_args.args[0])

    def test_report_keeps_chapters_and_leads_with_simple_summary(self):
        from report import build_sections
        snapshot = json.loads((ROOT / "data/reviewed_evidence.json").read_text())
        sources = read_sources()
        chunks = [{"source_id": s["id"]} for s in sources]
        state = {"market": [validate(x, "market", sources, chunks) for x in snapshot["market"]],
                 "domain": [validate(x, "domain", sources, chunks) for x in snapshot["domain"]],
                 "synthesis": snapshot["synthesis"], "sources": sources, "as_of": "2026-09-23",
                 "mode": "replay", "corpus_pages": 169, "embedding_model": "test"}
        sections = build_sections(state)
        self.assertEqual([heading for heading, _ in sections],
                         ["SUMMARY", "1. 분석 배경", "2. 기술 선정과 개요", "3. 평가 방법",
                          "4. 시장성 평가 / DeepSeek-V2 MLA", "4. 시장성 평가 / CXL-PNM (PNM-KV / PnG-KV)",
                          "5. 도메인 평가 / AI 코딩·업무 에이전트", "6. 시사점과 한계", "REFERENCE"])
        self.assertEqual([kind for kind, _ in sections[0][1]], ["p"])
        for _, blocks in sections[1:-1]:
            self.assertIn("lead", [kind for kind, _ in blocks])
            self.assertIn("table", [kind for kind, _ in blocks])


if __name__ == "__main__":
    unittest.main()
