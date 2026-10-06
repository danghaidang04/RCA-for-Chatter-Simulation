"""Trích TÍN HIỆU THÔ của cùng các cửa sổ 0.3 s (như real_features.py) ra output/real_windows.npz để ước lượng phổ/nhiễu."""
import glob, os, sys
import numpy as np, pandas as pd
from scipy.io import wavfile

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "msm", "imi_vm20i")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
W, NWIN, TRIM, G_MM = 0.3, 3, 0.15, 9810.0


def main(datasets, outfile="real_windows.npz"):
    A0, A1, M, meta = [], [], [], []
    fs_a_all = sr_all = None
    for ds in datasets:
        d = f"{ROOT}/dataset{ds}"; lab = pd.read_csv(f"{d}/label.csv")
        acc = pd.read_csv(glob.glob(f"{d}/*_acc.csv")[0], dtype=np.float32, engine="c")
        t = acc["time"].to_numpy(np.float64); a0 = acc["acc0"].to_numpy() / G_MM; a1 = acc["acc1"].to_numpy() / G_MM; fs_a = 1.0 / np.median(np.diff(t[:2000]))
        sr, w = wavfile.read(glob.glob(f"{d}/*_s0.wav")[0]); w = w.astype(np.float32) / 32768.0
        fs_a_all, sr_all = fs_a, sr
        for i, r in lab.iterrows():
            dur = r.end - r.start; s, e = r.start + TRIM * dur, r.end - TRIM * dur
            for k, t0 in enumerate(np.linspace(s, e - W, NWIN)):
                A0.append(a0[int(t0 * fs_a):int(t0 * fs_a) + int(W * fs_a)]); A1.append(a1[int(t0 * fs_a):int(t0 * fs_a) + int(W * fs_a)])
                M.append(w[int(t0 * sr):int(t0 * sr) + int(W * sr)])
                meta.append((ds, i, k, int(r.label), int(r.direction == "UP"), float(r.N), float(r.ae), float(r.F), int(r.Z)))
        print("dataset", ds, "xong", flush=True)
    n = min(map(len, A0)); m = min(map(len, M))
    meta = np.array(meta)
    np.savez_compressed(f"{OUT}/{outfile}", acc0=np.stack([x[:n] for x in A0]), acc1=np.stack([x[:n] for x in A1]), mic=np.stack([x[:m] for x in M]),
                        meta=meta, fs_a=fs_a_all, sr=sr_all)
    print("saved", meta.shape, n, m)


if __name__ == "__main__":
    args = sys.argv[1:]; out = args[0] if args and not args[0].isdigit() else "real_windows.npz"
    main([int(x) for x in args if x.isdigit()], out)
