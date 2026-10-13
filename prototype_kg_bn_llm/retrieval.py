"""Truy hồi kiểu LightRAG (KHÔNG dùng LLM để lập chỉ mục): đồ thị thực thể-quan hệ từ kho tài liệu nhỏ, truy hồi HAI TẦNG
 - tầng thấp: từ khóa khớp THỰC THỂ (tên thực thể là khóa duy nhất);  - tầng cao: từ khóa khớp QUAN HỆ/chủ đề;  rồi mở rộng MỘT bước láng giềng (các đoạn chung thực thể).
Chỉ dùng để LẬP LUẬN về thứ hạng do mạng Bayes tạo ra; không bao giờ ảnh hưởng đến thứ hạng. Đây là bản mô phỏng ý tưởng LightRAG, không phải thư viện lightrag."""
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__)); TOK = re.compile(r"[a-z0-9]+")
FEATURE_THEMES = {"log10_acc": ["vibration"], "mic_dB": ["noise"], "chatter_acc": ["chatter", "vibration"], "chatter_mic": ["chatter"], "acc_low": ["force", "vibration"], "acc_res": ["vibration", "stiffness"], "acc_high": ["vibration"], "mic_low": ["noise", "force"], "mic_high": ["noise"]}


def toks(s): return set(TOK.findall(s.lower()))


class Retriever:
    def __init__(self, path=None):
        self.entries = json.load(open(path or os.path.join(HERE, "documents", "corpus_demo.json")))["entries"]; self.by_id = {e["id"]: e for e in self.entries}
        self.ent = {}; self.edges = []                                          # thực thể -> các đoạn; danh sách quan hệ (e1, e2, khóa, đoạn)
        for e in self.entries:
            for n in e["entities"]: self.ent.setdefault(n.lower(), set()).add(e["id"])
            for a, b, k in e["relations"]: self.edges.append((a.lower(), b.lower(), toks(k), e["id"]))
        N = len(self.entries); df = {}
        for e in self.entries:
            for t in set().union(*[toks(n) for n in e["entities"]], *[toks(k) for _, _, k in e["relations"]], toks(" ".join(e["themes"]))): df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log((N + 1) / (c + 0.5)) + 1 for t, c in df.items()}

    def retrieve(self, low, high, top_n=4):
        """low: từ khóa tầng thấp (thực thể); high: từ khóa tầng cao (chủ đề/quan hệ). Trả về danh sách đoạn kèm lý do truy hồi."""
        lq = set().union(*[toks(x) for x in low]) if low else set(); hq = set().union(*[toks(x) for x in high]) if high else set(); sc = {}; why = {}
        for e in self.entries:
            es = set().union(*[toks(n) for n in e["entities"]]); s_low = sum(self.idf.get(t, 1) for t in lq & es)
            rs = set().union(*[toks(k) for _, _, k in e["relations"]]) | toks(" ".join(e["themes"])); s_high = sum(self.idf.get(t, 1) for t in hq & rs)
            if s_low + s_high > 0: sc[e["id"]] = s_low + 0.7 * s_high; why[e["id"]] = [f"low-level match: {sorted(lq & es)}" if lq & es else "", f"high-level match: {sorted(hq & rs)}" if hq & rs else ""]
        seeds = sorted(sc, key=sc.get, reverse=True)[:2]
        for sid in seeds:                                                       # mở rộng một bước: đoạn chung thực thể với đoạn hạt giống
            for n in self.by_id[sid]["entities"]:
                for nb in self.ent.get(n.lower(), ()):
                    if nb != sid and nb not in sc: sc[nb] = 0.3 * sc[sid]; why[nb] = [f"1-hop neighbour of {sid} via entity '{n}'", ""]
        out = []
        for i in sorted(sc, key=sc.get, reverse=True)[:top_n]:
            e = self.by_id[i]; out.append({"id": i, "source": e["source"], "url": e["url"], "text_type": e["text_type"], "quote": e["quote"], "cause_tags": e["cause_tags"], "score": round(sc[i], 3), "why": "; ".join(x for x in why[i] if x)})
        return out
