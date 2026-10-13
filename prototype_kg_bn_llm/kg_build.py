"""Dựng KG TỰ ĐỘNG từ tài liệu (kho documents/corpus_demo.json): trích bộ ba nguyên nhân -> cơ chế -> đặc trưng cảm biến, và nguyên nhân -> biện pháp, gộp kiểu LightRAG (hợp, loại trùng, đếm số đoạn ủng hộ).
 Chế độ 'rule' : từ vựng + quy tắc (xác định, không cần LLM). Mọi liên kết 'triệu chứng -> đặc trưng cảm biến' lấy từ BẢNG CẦU do người viết (tài liệu nói 'vibration', không nói 'acc_res'): đây là điểm yếu nhất, nói rõ trong báo cáo.
 Chế độ 'llm'  : xây prompt + lược đồ, kiểm tra đầu ra (nhãn thuộc ontology, trích dẫn là chuỗi con của đoạn); chạy qua OpenRouter chỉ khi có --send. Chưa chạy trong báo cáo hiện tại.
 Đầu ra: kg/auto_kg.json (cùng lược đồ với kg/milling_kg.json, tier ASSUMED), report/auto_kg_eval.json. Chạy: python kg_build.py [--llm --send]"""
import argparse
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAUSE_LEX = {"STIFFNESS": ["weak fixture", "fixture support", "workpiece rigidity", "axially weak workpiece", "clamping", "rigidity"], "DAMPING": ["tool overhang", "stick-out", "damping"],
             "TOOL_WEAR": ["flank wear", "crater wear", "tool wear", "chipping"], "DEPTH_OVERLOAD": ["cutting depth", "depth of cut"], "RPM_MISMATCH": ["cutting speed", "spindle speed"]}
SYMPTOM_FEATURES = {"vibration": ["log10_acc", "acc_res"], "chatter": ["chatter_acc", "chatter_mic"], "noise": ["mic_dB", "mic_high"], "force": ["acc_low", "mic_low"], "wear": ["mic_low", "acc_low"]}   # BẢNG CẦU viết tay
ONTOLOGY = {"entity_types": ["cause", "mechanism", "feature", "action"], "relations": ["leads_to", "manifests_as", "verified_by"], "causes": list(CAUSE_LEX), "features": sorted({f for v in SYMPTOM_FEATURES.values() for f in v})}


def extract_rule(entry):
    """Một đoạn -> danh sách bộ ba (src, rel, dst, bằng chứng). Nguyên nhân suy từ VĂN BẢN (nguồn, trích dẫn, thực thể), không dùng nhãn cause_tags."""
    text = " ".join([entry["source"], entry["quote"]] + entry["entities"]).lower(); causes = [c for c, ph in CAUSE_LEX.items() if any(p in text for p in ph)]; trip = []
    syms = [s for s in SYMPTOM_FEATURES if s in (" ".join(entry["themes"]) + " " + text)]
    for c in causes:
        mech = next((e for e in entry["entities"] if any(p == e.lower() or p in e.lower() for p in CAUSE_LEX[c])), entry["entities"][0])
        trip.append((c, "leads_to", f"M::{mech.lower()}", entry["id"]))
        for s in syms:
            for f in SYMPTOM_FEATURES[s]: trip.append((f"M::{mech.lower()}", "manifests_as", f, entry["id"]))
        trip.append((c, "verified_by", f"A::{entry['quote'][:80]}", entry["id"]))
    return trip


def build(entries, extractor):
    rel = {}
    for e in entries:
        for s, r, d, src in extractor(e): rel.setdefault((s, r, d), []).append(src)
    ents = {}; rels = []; by = {e["id"]: e for e in entries}
    for (s, r, d), srcs in rel.items():
        for n in (s, d):
            if n not in ents:
                kind = "cause" if n in ONTOLOGY["causes"] else ("feature" if n in ONTOLOGY["features"] else ("mechanism" if n.startswith("M::") else "action")); ents[n] = {"id": n, "type": kind, "label": n.split("::")[-1], "description": "auto-extracted"}
        rels.append({"id": f"ra{len(rels):03d}", "source": s, "target": d, "relation": r, "sign": 0, "rationale": "supported by " + ", ".join(sorted(set(srcs))), "provenance": {"tier": "ASSUMED", "reference": "; ".join(sorted({by[i]["url"] for i in srcs})), "support": sorted(set(srcs)), "text_types": sorted({by[i]["text_type"] for i in srcs})}})
    return {"meta": {"name": "auto-built KG (rule extractor)", "note": "built from documents/corpus_demo.json; symptom->feature links come from a hand-written bridge lexicon; all tier ASSUMED", "tiers": ["FIXED", "RANGE", "LEARNED", "ASSUMED"]}, "entities": list(ents.values()), "relations": rels}


def cause_feature_edges(kgd):
    mech = {}; out = {}
    for r in kgd["relations"]:
        if r["relation"] == "leads_to": mech.setdefault(r["target"], set()).add(r["source"])
    for r in kgd["relations"]:
        if r["relation"] == "manifests_as":
            for c in mech.get(r["source"], ()): out.setdefault((c, r["target"]), set()).update(r["provenance"]["support"])
    return out


def llm_prompt(entry):
    return ("Extract causal knowledge about milling from the passage. Return JSON {\"triples\":[{\"source\":..., \"relation\":..., \"target\":..., \"evidence\":...}]}. "
            f"Allowed relations: {ONTOLOGY['relations']}. 'source' of leads_to must be one of {ONTOLOGY['causes']} (or omit); 'target' of manifests_as must be one of {ONTOLOGY['features']}; "
            "'evidence' must be copied exactly from the passage. Use only the passage; do not add knowledge.\nPassage: " + entry["quote"] + "\nSource: " + entry["source"])


def validate_llm_triples(triples, entry):
    bad = []
    for t in triples:
        if t.get("relation") not in ONTOLOGY["relations"]: bad.append(f"relation {t.get('relation')}")
        if t.get("relation") == "manifests_as" and t.get("target") not in ONTOLOGY["features"]: bad.append(f"feature {t.get('target')}")
        if t.get("relation") == "leads_to" and t.get("source") not in ONTOLOGY["causes"]: bad.append(f"cause {t.get('source')}")
        if t.get("evidence") and t["evidence"] not in entry["quote"]: bad.append("evidence not in passage")
    return bad


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--llm", action="store_true"); ap.add_argument("--send", action="store_true"); a = ap.parse_args()
    if a.llm and not a.send: raise SystemExit("LLM extraction sends the passages to OpenRouter: add --send to confirm")
    if a.llm: raise SystemExit("LLM mode: prompt and validator are implemented (llm_prompt, validate_llm_triples) but the run is not wired yet; confirm with the supervisor/user first")
    entries = json.load(open(os.path.join(HERE, "documents", "corpus_demo.json")))["entries"]; kgd = build(entries, extract_rule); json.dump(kgd, open(os.path.join(HERE, "kg", "auto_kg.json"), "w"), indent=1)
    from kg import CAUSES, FEATURES, KnowledgeGraph
    auto = cause_feature_edges(kgd); hand = {k for k in KnowledgeGraph().compile_edges()}; cond = json.load(open(os.path.join(HERE, "report", "cond24_results_k10.json"))); et = cond["edges_table"]
    data = {tuple(k.split("|")) for k, e in et.items() if (e["ci_lo"] > 0 or e["ci_hi"] < 0) and abs(e["weight"]) >= 0.5}; A = set(auto)
    prec = lambda S, T: (len(S & T) / len(S)) if S else float("nan"); rec = lambda S, T: (len(S & T) / len(T)) if T else float("nan")
    res = {"passages": len(entries), "entities": len(kgd["entities"]), "relations": len(kgd["relations"]), "auto_cause_feature_edges": len(A), "hand_kg_edges": len(hand), "data_supported_edges": len(data),
           "auto_vs_hand": {"overlap": len(A & hand), "jaccard": len(A & hand) / len(A | hand), "precision": prec(A, hand), "recall": rec(A, hand)}, "auto_vs_data": {"precision": prec(A, data), "recall": rec(A, data), "n_auto_supported": len(A & data)},
           "hand_vs_data": {"precision": prec(hand, data), "recall": rec(hand, data)}, "causes_without_any_edge": [c for c in CAUSES if not any(e[0] == c for e in A)], "edges_support_count": {f"{c}|{f}": len(v) for (c, f), v in auto.items()}}
    json.dump(res, open(os.path.join(HERE, "report", "auto_kg_eval.json"), "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k != "edges_support_count"}, indent=1))


if __name__ == "__main__":
    main()
