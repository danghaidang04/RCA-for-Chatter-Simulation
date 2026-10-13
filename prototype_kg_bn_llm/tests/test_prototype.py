"""Kiểm thử: python -m unittest discover -s tests -v (chạy từ prototype_kg_bn_llm/). Không cần mạng, không cần khóa."""
import copy
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bn import CondGaussianBN, kg_mask, score_pagerank, shuffled_mask
from kg import CAUSES, FEATURES, KnowledgeGraph
from llm_reasoner import OpenRouterReasoner, TemplateReasoner, reason, validate_reasoning
import run_demo

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def toy_model():
    rng = np.random.default_rng(0); y = np.repeat(np.arange(6), 200); mu = rng.normal(0, 1.5, (6, 9)); mu[0] = 0; X = mu[y] + rng.normal(0, 1, (len(y), 9)); return CondGaussianBN().fit(X, y), mu


class TestKG(unittest.TestCase):
    def test_compiled_edges(self):
        kg = KnowledgeGraph(); e = kg.compile_edges(); self.assertEqual(len(e), 19)
        self.assertIn(("TOOL_WEAR", "mic_dB"), e); self.assertNotIn(("TOOL_WEAR", "acc_high"), e)
        M, S = kg_mask(kg); self.assertEqual(M.sum(), 19); self.assertEqual(shuffled_mask(M, 1).sum(axis=1).tolist(), M.sum(axis=1).tolist())

    def test_every_relation_has_provenance(self):
        for r in KnowledgeGraph().relations: self.assertIn(r["provenance"]["tier"], ["FIXED", "RANGE", "LEARNED", "ASSUMED"])


class TestBN(unittest.TestCase):
    def test_posterior_normalised_and_contributions_add_up(self):
        m, mu = toy_model(); D = mu[2] + np.random.default_rng(1).normal(0, 1, (10, 9)); idx, p = m.posterior(D); self.assertAlmostEqual(p.sum(), 1.0, places=9)
        ll = m.loglik(D); self.assertAlmostEqual(m.contributions(D, 2, ref=0).sum(), ll[2] - ll[0], places=6)

    def test_posterior_picks_true_state(self):
        m, mu = toy_model(); D = mu[3] + np.random.default_rng(2).normal(0, 1, (10, 9)); self.assertEqual(int(np.argmax(m.posterior(D)[1])), 3)

    def test_pagerank_is_distribution_like(self):
        r = score_pagerank(np.random.default_rng(0).normal(size=(5, 9)), np.random.default_rng(1).normal(size=(10, 9))); self.assertTrue(np.all(r >= 0))


def evidence():
    m, mu = toy_model(); kg = KnowledgeGraph(); D = mu[3] + np.random.default_rng(5).normal(0, 1, (10, 9)); edges = {(c, f): {"weight": 1.0, "ci_lo": 0.5, "ci_hi": 1.5} for c in CAUSES for f in FEATURES}
    consts = {"node_sd_ref": [1.0] * 9, "q95_nominal": 30.0}; return run_demo.build_evidence(m, kg, D, "T", "toy", consts, edges)


class TestReasoning(unittest.TestCase):
    def test_template_output_is_valid_and_ranking_untouched(self):
        ev = evidence(); out = reason(ev, TemplateReasoner()); self.assertTrue(out["validated"]); self.assertEqual(out["ranking"], ev["posterior"])

    def test_validator_rejects_fabricated_number_cause_and_decision(self):
        ev = evidence(); good = TemplateReasoner().explain(ev); self.assertEqual(validate_reasoning(good, ev), [])
        bad = copy.deepcopy(good); bad["summary"] += " The posterior is 0.9137 and CHIPPED_FLUTE is definitely present, so shut down the machine."
        v = validate_reasoning(bad, ev); self.assertTrue(any("0.9137" in x for x in v)); self.assertTrue(any("CHIPPED_FLUTE" in x for x in v)); self.assertTrue(any("decision" in x for x in v))

    def test_bad_backend_falls_back_to_template(self):
        class Evil:
            name = "evil"
            def explain(self, ev): return {"summary": "Nothing to see.", "supporting_evidence": [], "conflicts_or_unexplained": [], "caveats": [], "suggested_checks": []}
        out = reason(evidence(), Evil()); self.assertTrue(out["validated"]); self.assertIn("fallback", out["backend"]); self.assertTrue(out["llm_violations"])

    def test_openrouter_refuses_without_permission_or_key(self):
        with self.assertRaises(PermissionError): OpenRouterReasoner(allow_external=False, env={"OPENROUTER_API_KEY": "x"}).explain(evidence())
        with self.assertRaises(RuntimeError): OpenRouterReasoner(allow_external=True, env={}).explain(evidence())

    def test_no_secret_in_repo_files(self):
        for root, _, files in os.walk(HERE):
            if "report" in root and "llm_calls" in root: continue
            for f in files:
                if f.endswith((".py", ".json", ".md", ".example")):
                    t = open(os.path.join(root, f), errors="ignore").read(); self.assertNotRegex(t, r"sk-or-[A-Za-z0-9\-_]{10,}", msg=f)


if __name__ == "__main__":
    unittest.main()


class TestKGBuild(unittest.TestCase):
    def test_rule_extraction_uses_text_not_labels(self):
        import kg_build
        e = {"id": "RX", "source": "Test, vibration: weak fixture", "url": "u", "text_type": "verbatim", "quote": "Reduce the cutting depth", "entities": ["weak fixture", "cutting depth"], "relations": [], "themes": ["vibration"], "cause_tags": []}
        t = kg_build.extract_rule(e); causes = {x[0] for x in t if x[1] == "leads_to"}; self.assertEqual(causes, {"STIFFNESS", "DEPTH_OVERLOAD"}); self.assertTrue(any(x[1] == "manifests_as" and x[2] == "acc_res" for x in t))

    def test_llm_triple_validator_rejects_bad_labels_and_invented_evidence(self):
        import kg_build
        e = {"quote": "Minimize overhang"}; good = [{"source": "DAMPING", "relation": "leads_to", "target": "x", "evidence": "Minimize overhang"}]; self.assertEqual(kg_build.validate_llm_triples(good, e), [])
        bad = [{"source": "MAGIC", "relation": "leads_to", "target": "x", "evidence": "invented sentence"}, {"source": "a", "relation": "causes", "target": "b"}, {"relation": "manifests_as", "target": "not_a_feature"}]
        self.assertGreaterEqual(len(kg_build.validate_llm_triples(bad, e)), 4)

    def test_retrieved_citations_and_quotes_are_validated(self):
        ev = evidence(); ev["retrieved"] = [{"id": "R1", "quote": "Minimize overhang", "source": "s", "url": "u", "text_type": "verbatim", "why": "w"}]; good = TemplateReasoner().explain(ev); self.assertEqual(validate_reasoning(good, ev), [])
        bad = copy.deepcopy(good); bad["summary"] += " See [R9] and «invented words»."; v = validate_reasoning(bad, ev); self.assertTrue(any("R9" in x for x in v)); self.assertTrue(any("invented" in x for x in v))
