"""Chế độ vận hành của mô phỏng ở tham số DANH ĐỊNH: so giới hạn ổn định ZOA với độ sâu cắt thật (ap=38.1 mm) và biên độ miền thời gian.
Điều kiện lấy từ dữ liệu thật (N=9000 rpm, ae={0.0635,0.3175}); k_modal nhân hệ số kscale. Ghi output/regime_check.json."""
import json
import numpy as np
from model_v4 import build, X0
from common import run
from milling_model import stability_lobes_zoa, a_lim_at

res = []
for ks in (1, 3, 10, 30):
    th = X0.copy(); th[7] += np.log(ks)
    for ae in (0.0635, 0.3175):
        cut = dict(N=9000.0, ap=38.1, ae=ae, F=16.93, Z=2, D=6.35, direction="DOWN"); p, cfg = build(cut, th)
        r, a, _ = stability_lobes_zoa(p, fc_range=(300, 4000), n_fc=3000, rpm_max=20000, alim_max=0.2)
        al = float(a_lim_at(r, a, np.array([9000.0]))[0] * 1e3) if len(r) else None
        sim, sig, _ = run(p, seed=1, t_end=0.6, sensor_cfg=cfg, noise_um=0.02, skip=0.5)
        res.append({"kscale": ks, "ae_mm": ae, "zoa_alim_mm": al, "ap_mm": 38.1, "x_rms_um": float(np.std(sim["x"][-2048:]) * 1e6), "Fx_rms_N": float(np.std(sim["Fx"][-2048:])), "h_min": float(sim["h_min"])})
        print(res[-1])
json.dump(res, open("output/regime_check.json", "w"), indent=1)
