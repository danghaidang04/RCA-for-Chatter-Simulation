"""Trích đặc trưng từ dữ liệu thật MSM (Purdue, Sci. Data 2026) cho từng đường cắt trong label.csv.
Cùng định nghĩa đặc trưng với mô phỏng (features.py): acc RMS, mic level, chỉ số chatter phi đồng bộ.
 - acc.csv : acc0 (trục chính), acc1 [mm/s²] @ ~51.2 kHz  -> chuyển sang g
 - s0.wav  : mic trong buồng máy 48 kHz int16 (CHƯA hiệu chuẩn Pa -> mức tương đối dBFS, offset được học khi khớp)
Chạy: python real_features.py 5 6
"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy.io import wavfile
from features import nonsync_ratio

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "msm", "imi_vm20i")
G_MM = 9810.0


W = 0.3          # độ dài cửa sổ phân tích (s): PHẢI trùng với cửa sổ phân tích của mô phỏng (0.5·0.6 s)
NWIN = 3         # số cửa sổ lấy đều trên mỗi đường cắt


def cut_features(dataset, trim=0.15):
    d = f"{ROOT}/dataset{dataset}"
    lab = pd.read_csv(f"{d}/label.csv")
    acc = pd.read_csv(glob.glob(f"{d}/*_acc.csv")[0], dtype=np.float32, engine="c")
    t_acc = acc["time"].to_numpy(np.float64); a0 = acc["acc0"].to_numpy() / G_MM; a1 = acc["acc1"].to_numpy() / G_MM
    fs_a = 1.0 / np.median(np.diff(t_acc[:2000]))
    sr, w = wavfile.read(glob.glob(f"{d}/*_s0.wav")[0]); w = w.astype(np.float32) / 32768.0
    mtc = pd.read_csv(glob.glob(f"{d}/*_mtc.csv")[0], encoding="utf-8-sig")
    rows = []
    for i, r in lab.iterrows():
        dur = r.end - r.start; s, e = r.start + trim * dur, r.end - trim * dur
        sel = (mtc["time"] >= s) & (mtc["time"] <= e)
        rms = lambda v: float(np.sqrt(np.mean((v - v.mean()) ** 2)))
        for w_i, t0 in enumerate(np.linspace(s, e - W, NWIN)):
            ia = slice(int(t0 * fs_a), int((t0 + W) * fs_a)); iw = slice(int(t0 * sr), int((t0 + W) * sr))
            x0, x1, m = a0[ia], a1[ia], w[iw]
            rows.append(dict(dataset=dataset, cut=i, win=w_i, label=int(r.label), direction=r.direction, N=float(r.N), ap=r.ap, ae=r.ae, F=r.F,
                             Z=int(r.Z), D=r.D, tool=r.Tool, workpiece=r.Workpiece,
                             acc_rms_g=float(np.sqrt(rms(x0) ** 2 + rms(x1) ** 2)),
                             mic_db_fs=float(20 * np.log10(rms(m) + 1e-9)),
                             chatter_acc=nonsync_ratio(x0, fs_a, r.N, int(r.Z)), chatter_mic=nonsync_ratio(m, sr, r.N, int(r.Z)),
                             watt=float(mtc.loc[sel, "watt"].mean()), smarms=float(mtc.loc[sel, "smarms"].mean()),
                             s_load=float(mtc.loc[sel, "S_Axis_Load"].mean())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = pd.concat([cut_features(int(a)) for a in sys.argv[1:]], ignore_index=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "real_features.csv")
    out.to_csv(p, index=False); print(out.round(3).to_string()); print("saved", p)
