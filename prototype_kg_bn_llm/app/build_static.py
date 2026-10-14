"""Tạo bản tĩnh cho GitHub Pages (docs/): ghi sẵn các phiên mô phỏng thật (cùng mã với máy chủ) rồi để trang phát lại.
Chạy: cd prototype_kg_bn_llm && python -m app.build_static"""
import json, os, re, sys
from app import live, stream

HERE = os.path.dirname(os.path.abspath(__file__)); PROTO = os.path.dirname(HERE); OUT = os.path.join(os.path.dirname(PROTO), "docs")
RPM, AE, SEV, RAMP, ONSET, NF = 10500.0, 0.1905, 0.9, 6, 14, 48


def rec(fault):
    s = stream.Session(RPM, AE, False, 0, "template", False); steps = []
    for _ in range(ONSET): steps.append(s.step())
    if fault:
        s.apply(fault, SEV, RAMP, 1); steps = []
        for _ in range(NF): steps.append(s.step())
    else:
        for _ in range(30): steps.append(s.step())
    ex = None
    if fault and s.ev is not None: ex = s.explain()
    return {"steps": steps, "explain": ex}, s


def main():
    live.init(workers=6); healthy_s = stream.Session(RPM, AE, False, 0, "template", False); H = [healthy_s.step() for _ in range(ONSET + 30)]
    R = {"n_cal": stream.N_CAL, "n_diag": stream.N_DIAG, "onset": ONSET, "params": healthy_s.params, "healthy": H, "faults": {}, "in_model": {}}
    for f in [k for k in live.FAULTS if k != "HEALTHY"]:
        r, s = rec(f); R["faults"][f] = r; R["in_model"][f] = live.FAULTS[f][2]; det = next((i for i, o in enumerate(r["steps"]) if o.get("diagnosis")), None)
        print(f, "diagnosis at step", det, "top", (r["steps"][det]["diagnosis"]["top"] if det is not None else None), "alarm_tick", s.alarm_tick, flush=True)
    R["info"] = {"faults": {k: {"in_model": v[2], "range": v[1]} for k, v in live.FAULTS.items()}, "openrouter_key_present": False, "openrouter_model": "google/gemma-3-4b-it", "static": True}
    R["params0"] = live.nominal_params(); R["sld"] = {str(a): live.sld(None, a) for a in live.AE_VALUES}
    R["detector"] = json.load(open(os.path.join(HERE, "report", "detector.json")))
    os.makedirs(OUT, exist_ok=True); json.dump(R, open(os.path.join(OUT, "replay.json"), "w"), separators=(",", ":"))
    h = open(os.path.join(HERE, "static", "index.html")).read()
    api_old = h[h.index("async function api(path,opt)"):h.index("\n", h.index("async function api(path,opt)"))]
    api_new = """let REPLAY=null,ST={t:0,h:0,f:null,j:0,last:null};
async function api(path,opt){const R=REPLAY||(REPLAY=await (await fetch('replay.json')).json());
 if(path=='/api/info')return R.info;if(path=='/api/params')return R.params0;if(path=='/api/sld'){const b=JSON.parse(opt.body);const n=R.params0;if(['fn','zeta','k','Kt','Kr_ratio'].some(k=>Math.abs(b.params[k]/n[k]-1)>0.005))throw new Error('parameters are fixed in the static demo; run the app locally to change them');return R.sld[String(b.ae)]};if(path=='/api/detector')return R.detector;if(path=='/api/llm/test')return {ok:false,model:'',error:'Not available in the static demo. Run the app locally or on a server to use the LLM.'};
 if(path.endsWith('/start')){ST={t:0,h:0,f:null,j:0,last:null};return {sid:'static',n_cal:R.n_cal,n_diag:R.n_diag,params:R.params}}
 if(path.endsWith('/apply')){const b=JSON.parse(opt.body);if(ST.t<R.n_cal)throw new Error('wait until the healthy baseline is learned ('+R.n_cal+' windows)');ST.f=b.fault;ST.last=b.fault;ST.j=0;return {ok:true,tick:ST.t,final_value:0,in_model:R.in_model[b.fault]}}
 if(path.endsWith('/clear')){ST.f=null;return {ok:true}}
 if(path.endsWith('/explain')){const e=R.faults[ST.last]&&R.faults[ST.last].explain;if(!e)throw new Error('no diagnosis yet');return e}
 if(path.endsWith('/step')){let o;if(ST.f){const L=R.faults[ST.f].steps;if(ST.j<L.length)o=JSON.parse(JSON.stringify(L[ST.j]));else{o=JSON.parse(JSON.stringify(L[L.length-1]));delete o.diagnosis;delete o.collecting}ST.j++}
  else{const H=R.healthy;const i=ST.t<R.n_cal?ST.t:R.n_cal+(ST.h++%(H.length-R.n_cal));o=JSON.parse(JSON.stringify(H[i]));o.fault=null}
  o.tick=ST.t++;await new Promise(r=>setTimeout(r,200));return o}
 throw new Error('not available in the static demo')}"""
    h = h.replace(api_old, api_new)
    h = h.replace("<body>", '<body><div style="background:#fff7e0;color:#5c4400;padding:8px 16px;font:13px system-ui;border-bottom:1px solid #e6d28a">Static demo: this page replays <b>recorded simulator sessions</b> (10 500 rpm, a<sub>e</sub> = 0.1905 mm, fault severity 90 %, onset over 6 windows). Cutting condition, severity and the LLM backend are fixed here, and the LLM is not available; run the Python server from the repository for live simulation and OpenRouter. Simulated data only, not real machine data.</div>', 1)
    open(os.path.join(OUT, "live.html"), "w").write(open(os.path.join(HERE, "static", "index.html")).read())
    open(os.path.join(OUT, "index.html"), "w").write(h); print("written", OUT)


if __name__ == "__main__": main()
