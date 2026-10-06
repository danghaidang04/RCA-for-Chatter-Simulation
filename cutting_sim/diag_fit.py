import json
from calibrate_cma import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score

def main():
    cal = json.load(open(f"{OUT}/calibration_result.json")); th = np.array(list(cal["machine"].values())); sig = np.array(list(cal["sigma"].values()))
    df = pd.read_csv(f"{OUT}/real_features.csv"); d = df[df.label == 0].reset_index(drop=True); cuts = cuts_of(d); real = real_matrix(d)
    names = ["log10 acc", "mic dB", "chat_acc", "chat_mic"]
    with Pool(10) as pool:
        S = np.vstack([sim_features(cuts, th, None, 900 + r, pool, sig=sig) for r in range(5)])
    print("feature     real mean±sd      sim mean±sd")
    for j, n in enumerate(names):
        print(f"{n:10s} {real[:,j].mean():7.3f}±{real[:,j].std():.3f}   {S[:,j].mean():7.3f}±{S[:,j].std():.3f}")
    X = np.vstack([real, S]); y = np.r_[np.zeros(len(real)), np.ones(len(S))]; gg = np.r_[np.arange(len(real)), np.tile(np.arange(len(real)), 5)]
    for j in range(4):
        acc = cross_val_score(RandomForestClassifier(100, min_samples_leaf=3, random_state=0), X[:, [j]], y, groups=gg, cv=GroupKFold(5)).mean()
        print("C2ST single feature", names[j], round(acc, 3))
    # theo điều kiện: tương quan log acc với rpm / feed trong thật và sim
    for nm, M in (("real", real), ("sim", S[:len(real)])):
        print(nm, "corr(log acc, rpm)", np.corrcoef(M[:, 0], d.N)[0, 1].round(2), "corr(mic, rpm)", np.corrcoef(M[:, 1], d.N)[0, 1].round(2), "corr(log acc, F)", np.corrcoef(M[:, 0], d.F)[0, 1].round(2), "corr(log acc, ae)", np.corrcoef(M[:, 0], d.ae)[0, 1].round(2))
    print("unique rpm", sorted(d.N.unique()), "unique ae", sorted(d.ae.unique()))
if __name__ == "__main__":
    main()
