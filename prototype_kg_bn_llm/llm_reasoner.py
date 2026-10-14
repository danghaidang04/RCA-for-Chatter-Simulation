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
SECTIONS = ["ranking", "evidence", "conflicts", "checks", "caveats"]
SCHEMA = {"type": "object", "additionalProperties": False, "required": SECTIONS, "properties": {k: {"type": "string"} for k in SECTIONS}}
NAMES = {"HEALTHY": "healthy machine", "STIFFNESS": "loose clamp", "DAMPING": "low damping", "TOOL_WEAR": "tool wear", "DEPTH_OVERLOAD": "too much radial depth", "RPM_MISMATCH": "spindle speed error"}
SYSTEM_PROMPT = (
    "You explain the output of a Bayesian root-cause model for milling, like a careful research assistant writing a short briefing. You are NOT a decision maker. "
    "Write FIVE short paragraphs of flowing prose (2 to 5 sentences each, no bullet lists, no headings), one per JSON field: "
    "ranking = state the ranking of the three most likely causes with their probabilities exactly as in the evidence; "
    "evidence = explain which sensor bands support the top cause and how the knowledge-graph path links them (give the provenance tier), then cite the retrieved document passages as [R#] with any quoted words copied exactly inside angle quotes «...»; say these passages are generic guidance, not proof the cause occurred; if no passage was retrieved, say so; "
    "conflicts = say explicitly whether anything conflicts with the top cause or is unexplained (use the misfit block and the sign agreement of each path); if there is nothing, say that no conflict was found; "
    "checks = write the suggested checks as a short paragraph, using ONLY 'suggested_checks_from_kg'; if there are none, say so; "
    "caveats = state the limitations given in the evidence. "
    "Style: make it easy to scan. Use short sentences (under 25 words), put the conclusion first, name each sensor band exactly as in the evidence (for example acc_res), write probabilities as percentages with a % sign, avoid long parentheses and jargon, and do not repeat the same fact in two sections. Rules: the ranking and probabilities are final, never change or contradict them; use ONLY facts and numbers present in the evidence JSON, no new numbers, causes or mechanisms; "
    "do not give maintenance orders or say 'definitely'/'certainly'; cite only ids that exist in 'retrieved', each in its own brackets like [R1][R2], never grouped. Return JSON matching the schema.")
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
    v = []; keys = set(SECTIONS)
    if not isinstance(out, dict) or set(out) != keys or not all(isinstance(out[k], str) for k in keys): return ["output keys differ from the schema"]
    text = " ".join(out[k] for k in SECTIONS)
    facts = numeric_facts(evidence); allowed_ids = set(evidence["allowed_ids"]); ev_strings = " ".join(strings(evidence))
    for tok in set(re.findall(r"\b[A-Z][A-Z_0-9]{3,}\b", text)):
        if tok not in allowed_ids and tok not in {"KG", "BN", "LLM", "RCA", "FRF", "PSD", "ASSUMED", "FIXED", "RANGE", "LEARNED", "CI", "SD", "HEALTHY"}: v.append(f"unknown identifier: {tok}")
    for tok in NUM.findall(text):
        pct = tok.endswith("%"); x = float(tok.rstrip("%")); cand = [f * 100 for f in facts] if pct else facts + [f * 100 for f in facts if 0 <= abs(f) <= 1]       # chấp nhận cả phần trăm viết không kèm dấu %
        if tok in ev_strings: continue                                    # số xuất hiện nguyên văn trong chuỗi của bằng chứng (nhãn KG, đường truy vết)
        if not any(abs(abs(x) - abs(c)) <= (0.51 if (pct or abs(c) > 1.5) else 0.006 + 0.006 * abs(c)) for c in cand) and not (float(x).is_integer() and 0 <= abs(x) <= 20): v.append(f"number not in evidence: {tok}")
    rid = {r["id"] for r in evidence.get("retrieved", [])}; rq = [r["quote"] for r in evidence.get("retrieved", [])]
    for c in set(re.findall(r"R\d+", " ".join(re.findall(r"\[([^\]]*R\d+[^\]]*)\]", text)))):                  # cả dạng gộp [R1, R2]
        if c not in rid: v.append(f"citation to an unretrieved passage: {c}")
    for q in re.findall(r"«([^»]+)»", text):
        if not any(q in x for x in rq): v.append(f"quoted text not found in retrieved passages: {q[:40]}")
    if DECISION_WORDS.search(text): v.append("decision-making language")
    if evidence["top"] not in out["ranking"]: v.append("ranking paragraph does not mention the top-ranked cause")
    return v


class TemplateReasoner:
    """Quy tắc xác định: dựng lời giải thích dạng đoạn văn từ gói bằng chứng (KHÔNG phải LLM; dùng để kiểm thử và làm phương án dự phòng)."""
    name = "template"

    def explain(self, ev):
        top = ev["top"]; post = ev["posterior"]; nm = lambda c: f"{NAMES.get(c, c.lower())} ({c})"; mf = ev["misfit"]
        r = f"The Bayesian model ranks {nm(top)} first at {100 * post[0]['prob']:.0f}%"
        if len(post) > 1: r += f", followed by {nm(post[1]['cause'])} at {100 * post[1]['prob']:.0f}%"
        if len(post) > 2: r += f" and {nm(post[2]['cause'])} at {100 * post[2]['prob']:.0f}%"
        r += f". The probabilities are temperature-calibrated on m={ev['m']} windows from the same cutting condition. The ranking is final and comes from the Bayesian network, not from the text writer."
        if mf["flag"]: r += " The out-of-model check flags this input, so the ranking should be treated as unreliable here."
        sents = []
        for e in sorted(ev["kg_paths"], key=lambda e: -abs(e["llr"]))[:3]:
            sg = ("no sign is predicted by the knowledge graph" if e["sign_pred"] == 0 else ("its sign agrees with the knowledge graph" if e["sign_agrees"] else "its sign conflicts with the knowledge graph")) if e["in_kg"] else "this link is not in the knowledge graph and was suggested by the data"
            sents.append(f"The {e['node']} band moved {e['observed_shift']:+.2f} sd (learned weight {e['weight']:+.2f}, interval {e['ci_lo']:+.2f} to {e['ci_hi']:+.2f}); {sg}.")
        ev_p = " ".join(sents) if sents else "No sensor band carries a large contribution for the top cause."
        top_edges = [e for e in ev["kg_paths"] if e["in_kg"] and e["path_text"]]
        if top_edges:
            chain = []
            for seg in top_edges[0]["path_text"].split(" | "):
                for part in seg.split(" -> "):
                    part = part.split(" (")[0].strip()
                    if part and part not in chain: chain.append(part)
            ev_p += f" The knowledge graph links them as: {' -> '.join(chain)} (provenance tier {top_edges[0]['tier']})."
        rt = ev.get("retrieved", [])
        if rt:
            for x in rt[:3]: ev_p += f" Source [{x['id']}], {x['source']}, " + (f"states «{x['quote']}»." if x["text_type"] != "paraphrase" else f"gives related guidance (paraphrased: {x['quote']}).")
            ev_p += " These passages are generic guidance retrieved for this symptom pattern, not evidence that the cause occurred."
        else: ev_p += " No document passage was retrieved for this pattern, so there are no citations."
        conf = []
        if mf["flag"]: conf.append(f"the out-of-model residual {mf['d2']:.1f} exceeds the reference threshold {mf['threshold']:.1f}, so no hypothesis explains the data well")
        for u in mf["unexplained_nodes"][:3]: conf.append(f"the {u['node']} band stays unexplained (residual {u['z']:+.2f} reference s.d.)")
        bad = [e["node"] for e in ev["kg_paths"] if e["in_kg"] and e["sign_pred"] != 0 and not e["sign_agrees"]]
        if bad: conf.append("the observed sign of " + ", ".join(bad[:3]) + " conflicts with what the knowledge graph predicts")
        c_p = ("Conflicts found: " + "; ".join(conf) + ".") if conf else "No conflict was found: no band is left unexplained by the top hypothesis, the out-of-model check is within its reference range, and every band with a predicted sign agrees with the knowledge graph."
        chk = ev["suggested_checks_from_kg"].get(top, [])
        k_p = ("Suggested checks from the knowledge graph. " + " ".join(f"Check {i + 1}: {c.rstrip('.')}." for i, c in enumerate(chk))) if chk else "The knowledge graph records no specific check for this cause, so none is suggested here."
        return {"ranking": r, "evidence": ev_p, "conflicts": c_p, "checks": k_p, "caveats": " ".join(ev["limitations"][:2])}


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
                "response_format": {"type": "json_schema", "json_schema": {"name": "rca_explanation", "strict": True, "schema": SCHEMA}}, "provider": {"require_parameters": True}, "max_completion_tokens": 3500}
        req = urllib.request.Request(self.URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}", "X-Title": "milling-rca-kg-bn-prototype"})
        try:
            with urllib.request.urlopen(req, timeout=90) as r: resp = json.loads(r.read())
        except urllib.error.HTTPError as e: raise RuntimeError(f"OpenRouter HTTP {e.code}: {e.read()[:300]!r}")
        ch = resp["choices"][0]; content = ch["message"]["content"]
        try: out = json.loads(content, strict=False)
        except json.JSONDecodeError:
            if not getattr(self, "_retrying", False):                  # mô hình nhỏ đôi khi dừng giữa chuỗi: thử lại một lần
                self._retrying = True
                try: return self.explain(ev, case_id)
                finally: self._retrying = False
            raise RuntimeError(f"the model returned malformed JSON (finish_reason={ch.get('finish_reason')}, {len(content)} chars, ends with {content[-60:]!r})")
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
