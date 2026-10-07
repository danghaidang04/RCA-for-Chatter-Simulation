"""Kiểm chứng từng tài liệu qua Crossref + arXiv (độ giống tiêu đề), in ứng viên tốt nhất. Chạy: python verify_refs.py"""
import difflib, json, re, urllib.parse, urllib.request
Q = {
 "sim2real_chatter": "Transitioning from Simulation to Reality: Applying Chatter Detection Models to Real-World Machining Data",
 "sim_faults_budak": "Smart tool-related faults monitoring system using process simulation-based machine learning algorithms",
 "tl_chatter": "Transfer Learning for Autonomous Chatter Detection in Machining",
 "dat_wear": "Domain Adversarial Transfer Learning for Generalized Tool Wear Prediction",
 "single_sensor_wear": "Data-driven tool wear prediction in milling, based on a process-integrated single-sensor approach",
 "dt_pdm": "Digital Twin Technologies in Predictive Maintenance: Enabling Transferability via Sim-to-Real and Real-to-Sim Transfer",
 "luh_dataset": "Multivariate time series data of milling processes with varying tool wear and machine tools",
 "meta_dg_wear": "Semi-supervised multi-source meta domain generalization method for tool wear prediction",
 "phys_guided_wear": "Application of physics-guided deep learning model in tool wear monitoring of high-speed milling",
 "dt_mech_wear": "Digital twin with dynamic mechanistic simulation core for milling tool wear prediction",
 "hybrid_wear": "Hybrid data-driven and model-informed online tool wear detection in milling machining",
 "gan_stability": "Generative adversarial networks for milling stability prediction",
 "pinn_phm_review": "Physics-Informed Machine Learning in Prognostics and Health Management: A Systematic Literature Review",
 "causal_rca_mfg": "causal root cause analysis manufacturing process anomaly detection causal process graph",
}
def crossref(t):
    u="https://api.crossref.org/works?rows=3&select=title,author,issued,DOI,container-title,type&query.bibliographic="+urllib.parse.quote(t)
    try: return json.load(urllib.request.urlopen(u,timeout=30))["message"]["items"]
    except Exception as e: return []
def arxiv(t):
    u="http://export.arxiv.org/api/query?max_results=2&search_query="+urllib.parse.quote(f'ti:"{t[:90]}"')
    try: x=urllib.request.urlopen(u,timeout=30).read().decode()
    except Exception: return []
    out=[]
    for e in re.findall(r"<entry>(.*?)</entry>",x,re.S):
        out.append({"title":re.sub(r"\s+"," ",re.search(r"<title>(.*?)</title>",e,re.S).group(1)),"id":re.search(r"<id>(.*?)</id>",e).group(1),"year":re.search(r"<published>(\d{4})",e).group(1),"authors":re.findall(r"<name>(.*?)</name>",e)[:3]})
    return out
sim=lambda a,b: difflib.SequenceMatcher(None,a.lower(),b.lower()).ratio()
res={}
for k,t in Q.items():
    best=None
    for it in crossref(t):
        s=sim(t,it["title"][0]); 
        if not best or s>best[0]: best=(s,"crossref",it["title"][0],it.get("DOI"),it["issued"]["date-parts"][0][0],(it.get("author") or [{}])[0].get("family"),(it.get("container-title") or [""])[0][:40])
    for it in arxiv(t):
        s=sim(t,it["title"])
        if not best or s>best[0]+0.02: best=(s,"arxiv",it["title"],it["id"],int(it["year"]),it["authors"][0].split()[-1] if it["authors"] else None,"arXiv")
    res[k]=best; print(f"{k:20s} sim={best[0]:.2f} [{best[1]}] {best[2][:75]} | {best[5]} {best[4]} | {best[6]} | {best[3]}" if best else f"{k}: không thấy")
json.dump(res,open("verified_candidates.json","w"),indent=1,default=str)
