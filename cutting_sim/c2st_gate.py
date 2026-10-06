"""Cổng kiểm tra trước khi học nguyên nhân gốc: C2ST CHỈ trên dữ liệu NORMAL, với máy+σ đã khớp (A+C)."""
import json
from c2st import *


def main():
    cal = json.load(open(f"{OUT}/calibration_v2.json"))
    thA = np.array(list(cal["machine"].values())); sig = np.array(cal["sigma"])
    df = pd.read_csv(f"{OUT}/real_features.csv"); d = df[df.label == 0].reset_index(drop=True)
    cuts = cuts_of(d); real = real_matrix(d); cr = cond_matrix(cuts); grp = (d.dataset * 1000 + d.cut).values; out = {}
    with Pool(10) as pool:
        for name, th, s in (("uncalibrated", X0_M, np.zeros(7)), ("machine-fit (A)", thA, np.zeros(7)), ("machine+σ (A+C)", thA, sig)):
            S = np.vstack([sim_v2(cuts, th, None, 900 + r, pool, s) for r in range(R)])
            acc, sd_ = c2st(real, S, cr, grp); out[name] = {"c2st_acc": acc, "std": sd_}
            print(f"normal  {name:18s} C2ST acc = {acc:.3f} ± {sd_:.3f}", flush=True)
            if name.startswith("machine+σ"):
                Sm = S.reshape(R, len(real), 4).mean(0)
                print("   feature  real mean±sd | sim mean±sd")
                for j, n in enumerate(["log10 acc", "mic dB", "chat_acc", "chat_mic"]):
                    print(f"   {n:10s} {real[:, j].mean():7.3f}±{real[:, j].std():.3f} | {S[:, j].mean():7.3f}±{S[:, j].std():.3f}")
    json.dump(out, open(f"{OUT}/c2st_gate_normal.json", "w"), indent=1)


if __name__ == "__main__":
    main()
