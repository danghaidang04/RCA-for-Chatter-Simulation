"""Lớp LLM: CHỈ giải thích kết quả của mạng Bayes dựa trên KG; không đổi thứ hạng, không ra quyết định.

 evidence = gói bằng chứng có cấu trúc (hậu nghiệm, đóng góp từng nút, đường truy vết KG kèm tier nguồn gốc, độ lệch khỏi mô hình, kiểm tra gợi ý từ KG).
 Hai backend: TemplateReasoner (quy tắc, xác định, KHÔNG phải LLM) và OpenRouterReasoner (LLM thật, chỉ chạy khi người dùng bật cờ cho phép gửi dữ liệu).
 Mọi đầu ra đều qua validate_reasoning(); nếu vi phạm (số bịa, tên nguyên nhân lạ, ngôn ngữ ra quyết định) thì bị từ chối và dùng bản mẫu.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["summary", "supporting_evidence", "conflicts_or_unexplained", "caveats", "suggested_checks"],
          "properties": {k: ({"type": "string"} if k == "summary" else {"type": "array", "items": {"type": "string"}}) for k in ["summary", "supporting_evidence", "conflicts_or_unexplained", "caveats", "suggested_checks"]}}
SYSTEM_PROMPT = (
    "You explain the output of a Bayesian root-cause model for milling. You are NOT a decision maker. "
    "Rules: (1) The ranking and probabilities in the evidence are final; never change, re-rank or contradict them. "
    "(2) Use ONLY facts and numbers present in the evidence JSON; do not introduce new numbers, causes or mechanisms. "
    "(3) Explain which sensor nodes support the top cause, which knowledge-graph paths link them (quote the path text and its provenance tier), "
    "and which observations conflict with or are not explained by the model (use the misfit block). "
    "(4) State the limitations given in the evidence. (5) Suggested checks must come from 'suggested_checks_from_kg'. "
    "(6) Do not give maintenance orders or say 'definitely'/'certainly'. (7) You may cite passages from 'retrieved' only as [R#] with the exact id, and put any quoted words in angle quotes «...» copied exactly from the passage; "
    "retrieved passages are generic vendor guidance, not evidence that the cause occurred. Return JSON matching the schema.")
DECISION_WORDS = re.compile(r"\b(definitely|certainly|must replace|shut down|stop the machine|guaranteed|is certain)\b", re.I)


def load_env():
    """Đọc .env cục bộ (nếu có) vào dict; KHÔNG in giá trị."""
    p = os.path.join(HERE, ".env"); env = dict(os.environ)
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line: k, v = line.split("=", 1); env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return env


def numeric_facts(obj):
    out = []

    def walk(o):
        if isinstance(o, bool): return
        if isinstance(o, (int, float)): out.append(float(o))
        elif isinstance(o, dict): [walk(v) for v in o.values()]
        elif isinstance(o, (list, tuple)): [walk(v) for v in o]
    walk(obj); return out


def strings(obj):
    out = []

    def walk(o):
        if isinstance(o, str): out.append(o)
        elif isinstance(o, dict): [walk(v) for v in o.values()]
        elif isinstance(o, (list, tuple)): [walk(v) for v in o]
    walk(obj); return out


NUM = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?%?")


def validate_reasoning(out, evidence):
    """Trả về danh sách vi phạm (rỗng = hợp lệ)."""
    v = []; keys = set(SCHEMA["required"])
    if not isinstance(out, dict) or set(out) != keys: return ["output keys differ from the schema"]
    text = " ".join([out["summary"]] + sum([out[k] for k in keys if isinstance(out[k], list)], []))
    facts = numeric_facts(evidence); allowed_ids = set(evidence["allowed_ids"]); ev_strings = " ".join(strings(evidence))
    for tok in set(re.findall(r"\b[A-Z][A-Z_0-9]{3,}\b", text)):
        if tok not in allowed_ids and tok not in {"KG", "BN", "LLM", "RCA", "FRF", "PSD", "ASSUMED", "FIXED", "RANGE", "LEARNED", "CI", "SD", "HEALTHY"}: v.append(f"unknown identifier: {tok}")
    for tok in NUM.findall(text):
        pct = tok.endswith("%"); x = float(tok.rstrip("%")); cand = [f * 100 for f in facts] if pct else facts
        if tok in ev_strings: continue                                    # số xuất hiện nguyên văn trong chuỗi của bằng chứng (nhãn KG, đường truy vết)
        if not any(abs(abs(x) - abs(c)) <= (0.51 if pct else 0.006 + 0.006 * abs(c)) for c in cand) and not (float(x).is_integer() and 0 <= abs(x) <= 20): v.append(f"number not in evidence: {tok}")
    rid = {r["id"] for r in evidence.get("retrieved", [])}; rq = [r["quote"] for r in evidence.get("retrieved", [])]
    for c in set(re.findall(r"\[(R\d+)\]", text)):
        if c not in rid: v.append(f"citation to an unretrieved passage: {c}")
    for q in re.findall(r"«([^»]+)»", text):
        if not any(q in x for x in rq): v.append(f"quoted text not found in retrieved passages: {q[:40]}")
    if DECISION_WORDS.search(text): v.append("decision-making language")
    if evidence["top"] not in out["summary"]: v.append("summary does not mention the top-ranked cause")
    return v


class TemplateReasoner:
    """Quy tắc xác định: dựng lời giải thích từ gói bằng chứng (KHÔNG phải LLM; dùng để kiểm thử và làm phương án dự phòng)."""
    name = "template"

    def explain(self, ev):
        top = ev["top"]; p = {r["cause"]: r["prob"] for r in ev["posterior"]}; sup = []; conf = []
        s = f"The model ranks {top} first with posterior {p[top]:.2f} (temperature-calibrated, m={ev['m']} windows)."
        if ev["posterior"][1]["prob"] > 0.2: s += f" The runner-up is {ev['posterior'][1]['cause']} with {ev['posterior'][1]['prob']:.2f}."
        for e in ev["kg_paths"][:4]:
            sg = ("the knowledge graph predicts no sign for this edge" if e["sign_pred"] == 0 else ("the observed sign agrees with the knowledge graph" if e["sign_agrees"] else "the observed sign conflicts with the knowledge graph")) if e["in_kg"] else "this edge is not in the knowledge graph (suggested by the data)"
            sup.append(f"{e['node']}: observed shift {e['observed_shift']:+.2f} sd, contribution {e['llr']:+.2f}; edge weight {e['weight']:+.2f} [{e['ci_lo']:+.2f}, {e['ci_hi']:+.2f}]; {sg} ({e['path_text']}; tier {e['tier']}).")
        for r in ev.get("retrieved", [])[:3]:
            kind = "paraphrase" if r["text_type"] == "paraphrase" else "quote"; sup.append(f"Vendor guidance [{r['id']}] ({kind}; {r['source']}): «{r['quote']}». Retrieved because: {r['why']}. This is generic guidance, not evidence that the cause occurred.")
        mf = ev["misfit"]
        if mf["flag"]: s += " The out-of-model check flags this input, so the ranking should be treated as unreliable here."
        if mf["flag"]: conf.append(f"Out-of-model check: the Mahalanobis residual {mf['d2']:.1f} exceeds the reference threshold {mf['threshold']:.1f}; the data are not well explained by any hypothesis.")
        for u in mf["unexplained_nodes"][:4]: conf.append(f"Unexplained node {u['node']}: residual {u['z']:+.2f} reference s.d. after the best hypothesis.")
        if not conf: conf.append("No node is flagged as unexplained by the top hypothesis.")
        return {"summary": s, "supporting_evidence": sup or ["No knowledge-graph edge of the top cause carries a large contribution."], "conflicts_or_unexplained": conf, "caveats": list(ev["limitations"]), "suggested_checks": list(ev["suggested_checks_from_kg"].get(top, []))}


class OpenRouterReasoner:
    """Gọi LLM qua OpenRouter (API tương thích OpenAI). Chỉ chạy khi allow_external=True; khóa lấy từ biến môi trường hoặc .env; ghi nhật ký (không có khóa)."""
    name = "openrouter"
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, allow_external=False, model=None, env=None):
        self.env = load_env() if env is None else env; self.model = model or self.env.get("OPENROUTER_MODEL") or "google/gemma-3-4b-it"; self.allow = allow_external
        self.key = self.env.get("OPENROUTER_API_KEY", "")

    def explain(self, ev, case_id="case"):
        if not self.allow: raise PermissionError("external call not allowed: pass allow_external=True (CLI flag --send) after reviewing the evidence file")
        if not self.key: raise RuntimeError("OPENROUTER_API_KEY not set (environment or prototype_kg_bn_llm/.env)")
        body = {"model": self.model, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(ev, ensure_ascii=False)}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "rca_explanation", "strict": True, "schema": SCHEMA}}, "provider": {"require_parameters": True}, "max_completion_tokens": 1500}
        req = urllib.request.Request(self.URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}", "X-Title": "milling-rca-kg-bn-prototype"})
        try:
            with urllib.request.urlopen(req, timeout=90) as r: resp = json.loads(r.read())
        except urllib.error.HTTPError as e: raise RuntimeError(f"OpenRouter HTTP {e.code}: {e.read()[:300]!r}")
        content = resp["choices"][0]["message"]["content"]; out = json.loads(content)
        os.makedirs(os.path.join(HERE, "report", "llm_calls"), exist_ok=True)
        json.dump({"time": time.time(), "model": resp.get("model", self.model), "usage": resp.get("usage"), "request_messages": body["messages"], "response": out}, open(os.path.join(HERE, "report", "llm_calls", f"{case_id}_{int(time.time())}.json"), "w"), indent=1)
        return out


def reason(ev, backend=None, case_id="case"):
    """Chạy backend, kiểm tra; nếu vi phạm thì dùng bản mẫu. Thứ hạng luôn là của mạng Bayes (không lấy từ LLM)."""
    tpl = TemplateReasoner(); used = "template"; violations = []; out = None
    if backend is not None and getattr(backend, "name", "") != "template":
        try:
            out = backend.explain(ev, case_id=case_id) if isinstance(backend, OpenRouterReasoner) else backend.explain(ev); violations = validate_reasoning(out, ev); used = backend.name
        except (PermissionError, RuntimeError, ValueError, KeyError, urllib.error.URLError) as e: violations = [f"backend failed: {type(e).__name__}: {str(e)[:120]}"]; out = None
    if out is None or violations:
        rejected = used if out is not None else None; out = tpl.explain(ev); used = "template" + (f" (fallback; rejected {rejected})" if rejected else ""); violations_after = validate_reasoning(out, ev)
    else: violations_after = []
    return {"ranking": ev["posterior"], "explanation": out, "backend": used, "llm_violations": violations, "final_violations": violations_after, "validated": not violations_after}
