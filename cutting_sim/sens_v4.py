"""Phân tích độ nhạy (one-at-a-time quanh 3 điểm gốc ngẫu nhiên): mỗi tham số học được quét 5 mức trong cận,
đo thay đổi của TRUNG BÌNH theo nhóm điều kiện của 4 đặc trưng, chuẩn hóa bằng độ lệch chuẩn thật. Biến nào gần như không ảnh hưởng -> cố định."""
import json
from model_v4 import *
from calibrate_v3 import groups, group_stats


def main():
    df = pd.read_csv(f"{OUT}/real_features.csv"); dn = df[df.label == 0].reset_index(drop=True)
    cuts = cuts_of(dn); Rn = real_matrix(dn); gid = groups(dn); G = gid.max() + 1; s_j = Rn.std(0)
    rng = np.random.default_rng(0); bases = [X0] + [LO + rng.uniform(0.25, 0.75, len(LO)) * (HI - LO) for _ in range(2)]
    eff = np.zeros((len(bases), len(LO), 4))
    with Pool(10) as pool:
        for b, base in enumerate(bases):
            for i in range(len(LO)):
                mus = []
                for lv in np.linspace(0.05, 0.95, 5):
                    th = base.copy(); th[i] = LO[i] + lv * (HI[i] - LO[i])
                    mus.append(group_stats(sim4(cuts, th, 0, pool), gid, G)[0])
                mus = np.array(mus)                                   # (5, G, 4)
                eff[b, i] = np.sqrt(((mus.max(0) - mus.min(0)) ** 2).mean(0)) / s_j      # RMS (qua nhóm) của biên độ thay đổi / sd thật
            print("base", b, "xong", flush=True)
    S = eff.mean(0)
    out = pd.DataFrame(S, index=NAMES, columns=["log10 acc", "mic dB", "chat_acc", "chat_mic"]); out["max"] = out.max(1)
    out["tier"] = np.where(out["max"] > 0.5, "ảnh hưởng mạnh", np.where(out["max"] > 0.15, "vừa", "yếu"))
    out.round(3).to_csv(f"{OUT}/sensitivity_v4.csv"); print(out.round(3).to_string())


if __name__ == "__main__":
    main()
