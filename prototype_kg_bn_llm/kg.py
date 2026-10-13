"""Knowledge graph (KG): đọc, kiểm tra và biên dịch thành các cạnh nguyên nhân -> đặc trưng cho mạng Bayes.
KG chứa cả cơ chế trung gian (để TRUY VẾT và giải thích) và hành động kiểm tra; mạng Bayes chỉ dùng các cạnh trực tiếp cause -> feature được biên dịch
(nút trung gian là biến ẩn không nhận dạng được từ dữ liệu này)."""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CAUSES = ["STIFFNESS", "DAMPING", "TOOL_WEAR", "DEPTH_OVERLOAD", "RPM_MISMATCH"]
FEATURES = ["log10_acc", "mic_dB", "chatter_acc", "chatter_mic", "acc_low", "acc_res", "acc_high", "mic_low", "mic_high"]


class KnowledgeGraph:
    def __init__(self, path=None):
        self.path = path or os.path.join(HERE, "kg", "milling_kg.json"); d = json.load(open(self.path))
        self.meta = d["meta"]; self.entities = {e["id"]: e for e in d["entities"]}; self.relations = d["relations"]; self.validate()
        self.out = defaultdict(list)
        for r in self.relations: self.out[r["source"]].append(r)

    def validate(self):
        ids = set(self.entities); tiers = set(self.meta["tiers"])
        for r in self.relations:
            assert r["source"] in ids and r["target"] in ids, f"relation {r['id']}: unknown entity"
            assert r["provenance"]["tier"] in tiers, f"relation {r['id']}: unknown tier"
            assert r["sign"] in (-1, 0, 1), f"relation {r['id']}: sign"
        assert [c for c in CAUSES if c not in ids] == [] and [f for f in FEATURES if f not in ids] == []

    def paths(self, cause, feature, max_len=4):
        """Mọi đường đi cause -> ... -> feature qua các quan hệ leads_to / manifests_as (danh sách quan hệ)."""
        res = []

        def dfs(node, trail):
            if len(trail) > max_len: return
            for r in self.out.get(node, []):
                if r["relation"] not in ("leads_to", "manifests_as"): continue
                if r["target"] == feature: res.append(trail + [r])
                elif self.entities[r["target"]]["type"] == "mechanism": dfs(r["target"], trail + [r])
        dfs(cause, []); return res

    def compile_edges(self):
        """Cạnh trực tiếp cause -> feature: {(cause, feature): dict(sign, paths, tier)}. Dấu = tích các dấu dọc đường (0 nếu có dấu chưa xác định)."""
        edges = {}
        for c in CAUSES:
            for f in FEATURES:
                ps = self.paths(c, f)
                if not ps: continue
                signs = set()
                for p in ps:
                    s = 1
                    for r in p: s *= r["sign"]
                    signs.add(s)
                sign = signs.pop() if len(signs) == 1 else 0
                tier = "ASSUMED" if any(r["provenance"]["tier"] == "ASSUMED" for p in ps for r in p) else ps[0][0]["provenance"]["tier"]
                edges[(c, f)] = {"sign": sign, "paths": [[r["id"] for r in p] for p in ps], "tier": tier}
        return edges

    def explain_edge(self, cause, feature):
        """Văn bản truy vết một cạnh: chuỗi cơ chế, lý do và nguồn gốc (tier)."""
        out = []
        for p in self.paths(cause, feature):
            steps = [f"{self.entities[r['source']]['label']} -> {self.entities[r['target']]['label']} ({r['rationale']}; tier {r['provenance']['tier']}, ref: {r['provenance']['reference']})" for r in p]
            out.append(" | ".join(steps))
        return out

    def checks_for(self, cause):
        return [self.entities[r["target"]]["label"] for r in self.out.get(cause, []) if r["relation"] == "verified_by"]


if __name__ == "__main__":
    kg = KnowledgeGraph(); e = kg.compile_edges(); print(len(kg.entities), "entities,", len(kg.relations), "relations,", len(e), "compiled cause->feature edges")
    for (c, f), v in e.items(): print(f"{c:15s} -> {f:12s} sign {v['sign']:+d} tier {v['tier']}")
