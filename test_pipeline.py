"""Run: python -m unittest -v. No network, model, or API required."""
import copy
import json
import unittest
from unittest.mock import patch, Mock

from agents import validate, ask, Plan, evaluate
from retrieval import ROOT, read_sources
from scoring import growth, adoption, ecosystem, capacity, responsiveness


class RubricTests(unittest.TestCase):
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

    def test_response_missing_and_failure_are_distinct(self):
        self.assertEqual(responsiveness(None, None, None, None)["label"], "판정 유보")
        self.assertEqual(responsiveness(None, None, 0, None)["label"], "L0")
        self.assertEqual(responsiveness(0, 0, 5, True)["label"], "L3")
        self.assertEqual(responsiveness(0, 10, 5, True)["label"], "L2")
        self.assertEqual(responsiveness(0, 10.01, 5, True)["label"], "L1")
        self.assertEqual(responsiveness(-5, -5, 5, False)["label"], "L0")
        self.assertEqual(responsiveness(0, 0, -200, True)["label"], "L0")

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


if __name__ == "__main__":
    unittest.main()
