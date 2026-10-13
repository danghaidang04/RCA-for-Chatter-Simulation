"""Demo đầu-cuối: KG -> mạng Bayes -> gói bằng chứng -> giải thích (mẫu hoặc LLM qua OpenRouter) -> báo cáo.
 python run_demo.py                           # backend mẫu (không gửi gì ra ngoài)
 python run_demo.py --llm openrouter --send   # gửi gói bằng chứng (đã lưu ở report/evidence_*.json) tới OpenRouter; cần OPENROUTER_API_KEY trong môi trường hoặc .env"""
import argparse
import csv
import json
import os

import numpy as np

import experiment as ex
from bn import STATES, CondGaussianBN
from kg import CAUSES, FEATURES, KnowledgeGraph
from llm_reasoner import OpenRouterReasoner, TemplateReasoner, reason

HERE = os.path.dirname(os.path.abspath(__file__))
LIMITS = ["Edge weights and the covariance are learned from simulated windows, so they inherit the simulator's assumptions (nominal parameters above the stability limit, assumed fault severities and gains).",
          "Knowledge-graph edges and signs are physics reasoning with provenance tier ASSUMED, tested only against simulated data.",
          "The posterior is temperature-calibrated on simulated validation draws; calibration on real data has not been tested.",
          "No real multi-cause labelled data exist, so the ranking has not been validated on real cuts."]


def read_edges():
    return {(r["cause"], r["feature"]): r for r in csv.DictReader(open(os.path.join(HERE, "report", "edges.csv")))}


def build_evidence(model, kg, D, case_id, variant, consts, edges, retriever=None):
    m = len(D); idx, p = model.posterior(D); order = np.argsort(-p); post = [{"cause": STATES[idx[i]], "prob": float(p[i]), "rank": r + 1} for r, i in enumerate(order)]
    top = post[0]["cause"]; ct = STATES.index(top); xbar = D.mean(0); kge = kg.compile_edges(); paths = []
    if ct > 0:
        llr = model.contributions(D, ct, ref=0)
        for k in np.argsort(-np.abs(llr))[:5]:
            f = FEATURES[k]; e = edges[(top, f)]; sp = kge.get((top, f), {}).get("sign", None)
            txt = " || ".join(kg.explain_edge(top, f)) if (top, f) in kge else "not in the knowledge graph (data-suggested edge; expert review needed)"
            paths.append({"node": f, "observed_shift": float(xbar[k]), "llr": float(llr[k]), "weight": float(e["weight"]), "ci_lo": float(e["ci_lo"]), "ci_hi": float(e["ci_hi"]), "in_kg": (top, f) in kge,
                          "sign_pred": 0 if sp is None else int(sp), "sign_agrees": bool(sp in (None, 0) or np.sign(xbar[k]) == sp), "path_text": txt, "tier": kge[(top, f)]["tier"] if (top, f) in kge else "ASSUMED"})
    d2 = float(m * (xbar - model.mu[ct]) @ model.P @ (xbar - model.mu[ct])); res = (xbar - model.mu[ct]) / np.array(consts["node_sd_ref"]); unexpl = [{"node": FEATURES[k], "z": float(res[k])} for k in np.argsort(-np.abs(res)) if abs(res[k]) > 3][:5]
    retrieved = []
    if retriever is not None and ct > 0:
        from retrieval import FEATURE_THEMES
        mech = [kg.entities[r["target"]]["label"] for r in kg.out[top] if r["relation"] == "leads_to"]; low = [kg.entities[top]["label"]] + mech + [kg.entities[p["node"]]["label"] for p in paths[:3]]
        high = sorted({t for p in paths[:3] for t in FEATURE_THEMES[p["node"]]} | {"vibration"})
        if d2 > consts["q95_nominal"]: low += ["spindle", "axis", "tool run-out"]; high += ["machine", "diagnosis", "runout"]      # đầu vào ngoài mô hình: mở rộng sang lỗi máy
        retrieved = retriever.retrieve(low, high, top_n=4)
    ids = list(kg.entities) + ["HEALTHY"]
    return {"case_id": case_id, "variant": variant, "m": m, "temperature": float(model.T), "posterior": post, "top": top, "node_shifts": {FEATURES[k]: float(xbar[k]) for k in range(9)}, "kg_paths": paths,
            "misfit": {"d2": d2, "threshold": float(consts["q95_nominal"]), "flag": bool(d2 > consts["q95_nominal"]), "unexplained_nodes": unexpl},
            "suggested_checks_from_kg": {c["cause"]: kg.checks_for(c["cause"]) for c in post[:2] if c["cause"] != "HEALTHY"}, "retrieved": retrieved, "limitations": LIMITS, "allowed_ids": ids}


def md(case, ev, out):
    L = [f"# Case {case['id']}: {case['title']}", "", f"Model variant `{ev['variant']}`, m={ev['m']} windows, temperature T={ev['temperature']:.1f}. Truth: {case['truth']}.", "", "## Ranking (from the Bayesian network; final)", "", "| rank | state | posterior |", "|---|---|---|"]
    L += [f"| {r['rank']} | {r['cause']} | {r['prob']:.3f} |" for r in ev["posterior"]]
    mf = ev["misfit"]; L += ["", f"Out-of-model check: d²={mf['d2']:.1f} vs threshold {mf['threshold']:.1f} -> {'FLAGGED' if mf['flag'] else 'within the reference range'}.", "", f"## Explanation (backend: {out['backend']}; validated: {out['validated']})", "", out["explanation"]["summary"], "", "**Supporting evidence**", ""]
    L += [f"- {s}" for s in out["explanation"]["supporting_evidence"]] + ["", "**Conflicts or unexplained**", ""] + [f"- {s}" for s in out["explanation"]["conflicts_or_unexplained"]] + ["", "**Caveats**", ""] + [f"- {s}" for s in out["explanation"]["caveats"]] + ["", "**Suggested checks (from the KG)**", ""] + [f"- {s}" for s in out["explanation"]["suggested_checks"]]
    if out["llm_violations"]: L += ["", f"LLM output rejected: {out['llm_violations']}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="kg_validated"); ap.add_argument("--llm", default="template", choices=["template", "openrouter"]); ap.add_argument("--send", action="store_true", help="allow sending the evidence bundle to OpenRouter")
    ap.add_argument("--model", default=None); a = ap.parse_args()
    res = json.load(open(os.path.join(HERE, "report", "experiment_results.json"))); Xn, Xs, split, _ = ex.load_sim(); mask = np.array(res["masks"][a.variant]); model = ex.fit_variant(Xn, split, mask); kg = KnowledgeGraph(); edges = read_edges(); consts = res["misfit"]
    real = np.load(os.path.join(HERE, "data", "real_nodes.npz")); te = np.where(split == "test")[0]; rng = np.random.default_rng(3)
    cases = [{"id": "A", "title": "simulated cut, nominal generator", "truth": "DAMPING (simulated)", "D": Xn[rng.choice(te, ex.M_DRAW), 2, :]},
             {"id": "B", "title": "simulated cut, shifted generator", "truth": "TOOL_WEAR (simulated, physical parameters shifted)", "D": Xs[rng.choice(te, ex.M_DRAW), 3, :]},
             {"id": "C", "title": "real intentionally defective tool D*", "truth": "defective tool (cause not in the model's label set)", "D": real["z_bad"][rng.choice(len(real["z_bad"]), ex.M_DRAW, replace=False)]},
             {"id": "D", "title": "real healthy tool D", "truth": "healthy", "D": real["z_healthy"][rng.choice(len(real["z_healthy"]), ex.M_DRAW, replace=False)]}]
    backend = OpenRouterReasoner(allow_external=a.send, model=a.model) if a.llm == "openrouter" else TemplateReasoner(); rows = []
    for c in cases:
        ev = build_evidence(model, kg, c["D"], c["id"], a.variant, consts, edges); json.dump(ev, open(os.path.join(HERE, "report", f"evidence_{c['id']}.json"), "w"), indent=1)
        out = reason(ev, backend, case_id=c["id"]); json.dump(out, open(os.path.join(HERE, "report", f"case_{c['id']}.json"), "w"), indent=1)
        open(os.path.join(HERE, "report", f"case_{c['id']}.md"), "w").write(md(c, ev, out)); top = ev["posterior"][0]
        rows.append(f"| {c['id']} | {c['title']} | {top['cause']} ({top['prob']:.2f}) | {'FLAGGED' if ev['misfit']['flag'] else 'ok'} (d²={ev['misfit']['d2']:.0f}) | {out['backend']} | {out['validated']} |"); print(rows[-1], flush=True)
    open(os.path.join(HERE, "report", "demo_table.md"), "w").write("| case | description | top-1 (posterior) | out-of-model check | explanation backend | validated |\n|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
