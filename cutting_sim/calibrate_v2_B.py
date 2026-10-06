"""Chạy riêng giai đoạn B (mô hình nguyên nhân) từ checkpoint A+C đã lưu trong output/calibration_v2.json. Cấu hình rút gọn: 1 hạt giống, ≤40 thế hệ."""
import json
from calibrate_v2 import *


def main(maxiter=40, popsize=12):
    cal = json.load(open(f"{OUT}/calibration_v2.json")); th = np.array(list(cal["machine"].values())); sig = np.array(cal["sigma"])
    df = pd.read_csv(f"{OUT}/real_features.csv"); df["age"] = tool_age(df)
    dn, da = df[df.label == 0].reset_index(drop=True), df[df.label == 1].reset_index(drop=True)
    Rn, Ra = real_matrix(dn), real_matrix(da); scale = Rn.std(0) + 1e-6
    cn, ca = cond_matrix(cuts_of(dn)), cond_matrix(cuts_of(da)); cs = np.vstack([cn, ca]).std(0) + 1e-6
    cuta = cuts_of(da); ages = da.age.values; decB = lambda u: LO_B + u * (HI_B - LO_B)
    with Pool(10) as pool:
        def fB(u):
            thB = decB(u); rng = np.random.default_rng(1)
            return mmd2(sim_v2(cuta, th, fault_list(thB, ages, rng), 1, pool, sig), Ra, scale, ca, ca, cs)
        uB0 = (np.array([0.5, 1.0, 0.3, 1.0, 0.3, 0.3, 0.05] + [0.0] * 6) - LO_B) / (HI_B - LO_B); lossB0 = fB(uB0)
        ubB, fbB, hB, solsB = best_of(fB, uB0, maxiter, popsize, "[B faults]", seeds=(1,)); thB = decB(ubB)
        w = np.exp(thB[7:13] - thB[7:13].max()); w /= w.sum()
        print("FAULTS", dict(zip(NAMES_B[:7], np.round(thB[:7], 3))), "weights", dict(zip(CAUSE_ORDER, np.round(w, 3))), f"loss {lossB0:.4f} -> {fbB:.4f}", flush=True)
        cal.update(faults=dict(zip(NAMES_B, thB.tolist())), faults_theta=thB.tolist(), fault_weights=dict(zip(CAUSE_ORDER, w.tolist())), faults_loss=[lossB0, fbB], hist_B=hB)
        save(cal)
        Sn = sim_v2(cuts_of(dn), th, None, 5, pool, sig); Sa = sim_v2(cuta, th, fault_list(thB, ages, np.random.default_rng(6)), 6, pool, sig)
    np.savez(f"{OUT}/calibration_v2_features.npz", Rn=Rn, Ra=Ra, Sn=Sn, Sa=Sa, age=ages)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
