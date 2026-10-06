import json
from c2st import *

def main():
    cal = json.load(open(f"{OUT}/calibration_v2.json")); th = np.array(list(cal["machine"].values())); sig = np.array(cal["sigma"])
    df = pd.read_csv(f"{OUT}/real_features.csv"); d = df[df.label == 0].reset_index(drop=True); cuts = cuts_of(d); real = real_matrix(d)
    with Pool(10) as pool:
        S = np.mean([sim_v2(cuts, th, None, 900 + r, pool, sig) for r in range(5)], axis=0)       # trung bình 5 lần -> phần xác định theo điều kiện
        S1 = sim_v2(cuts, th, None, 901, pool, sig)
    names = ["log10 acc", "mic dB", "chat_acc", "chat_mic"]
    d["g"] = d.N.astype(int).astype(str) + "_" + d.ae.astype(str) + "_" + d.F.round(1).astype(str) + "_" + d.direction
    print("số nhóm điều kiện duy nhất:", d.g.nunique(), " số cửa sổ:", len(d))
    for j, n in enumerate(names):
        r_df = pd.DataFrame({"g": d.g, "r": real[:, j], "s": S1[:, j], "sm": S[:, j]})
        within_r = r_df.groupby("g").r.std().mean(); within_s = r_df.groupby("g").s.std().mean()
        mean_gap = (r_df.groupby("g").r.mean() - r_df.groupby("g").sm.mean()).abs().mean()
        print(f"{n:10s} within-group sd: real {within_r:.3f} sim {within_s:.3f} | mean |real-sim| per group {mean_gap:.3f} | real sd overall {real[:, j].std():.3f}")
    for c in ("N", "ae", "F"):
        print(c, "corr(log acc) real %.2f sim %.2f | corr(mic) real %.2f sim %.2f" % (np.corrcoef(real[:, 0], d[c])[0, 1], np.corrcoef(S1[:, 0], d[c])[0, 1], np.corrcoef(real[:, 1], d[c])[0, 1], np.corrcoef(S1[:, 1], d[c])[0, 1]))
    print("direction DOWN vs UP mean log acc: real %.3f/%.3f sim %.3f/%.3f" % (real[d.direction == "DOWN", 0].mean(), real[d.direction == "UP", 0].mean(), S1[d.direction == "DOWN", 0].mean(), S1[d.direction == "UP", 0].mean()))

if __name__ == "__main__":
    main()
