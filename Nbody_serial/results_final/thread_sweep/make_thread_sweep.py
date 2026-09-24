"""Kernel ablation vs OpenMP threads: N=10000, 5 steps, P=1, T=1,2,4,8,16.
T=1 from ../ablation_complete_20260922_083515, T>1 from ../ablation_T*_20260923_165910."""
import csv, glob, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []
for f in glob.glob("../ablation_T*_20260923_165910/ablation.csv") + ["../ablation_complete_20260922_083515/ablation.csv"]:
    rows += list(csv.DictReader(open(f)))
g = {}
for r in rows:
    g.setdefault((int(r["threads"]), r["Test_Type"], r["Config"]), []).append(r)
Ts = sorted({k[0] for k in g})
m = lambda T, t, c, k: st.median(float(x[k]) for x in g[(T, t, c)])
s = lambda T, t, c, k: st.stdev(float(x[k]) for x in g[(T, t, c)])
out = []
nf1 = m(1, "Kernel", "newton", "force")
for T in Ts:
    o = {"threads": T}
    for t, c in [("Kernel", "direct"), ("Kernel", "newton"), ("Math", "exact"), ("Math", "approx1"), ("Math", "approx2"),
                 ("Accumulators", "1"), ("Accumulators", "2"), ("Accumulators", "4"), ("Accumulators", "8")]:
        o[f"{t}_{c}_total_median"] = m(T, t, c, "Time_Sec")
        o[f"{t}_{c}_total_stdev"] = s(T, t, c, "Time_Sec")
        o[f"{t}_{c}_force_median"] = m(T, t, c, "force")
    o["newton_over_direct_force"] = o["Kernel_newton_force_median"] / o["Kernel_direct_force_median"]
    o["newton_force_model_static_imbalance"] = nf1 / T * (2 - 1 / T)
    o["approx1_force_speedup"] = o["Math_exact_force_median"] / o["Math_approx1_force_median"]
    o["approx2_force_speedup"] = o["Math_exact_force_median"] / o["Math_approx2_force_median"]
    o["all_ok"] = all(x["status"] == "OK" for k, v in g.items() if k[0] == T for x in v)
    out.append(o)
with open("thread_sweep_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

C1, C2 = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": "#8a8a85",
                     "axes.labelcolor": "#333", "xtick.color": "#555", "ytick.color": "#555"})
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))
d = [o["Kernel_direct_force_median"] for o in out]; n = [o["Kernel_newton_force_median"] for o in out]
a.plot(Ts, [d[0] / T for T in Ts], ls="--", color="#8a8a85", lw=1.2, label="direct, ideal")
a.plot(Ts, d, marker="o", ms=8, lw=2, color=C1, label="direct (every pair twice)", markeredgecolor="white", markeredgewidth=1.2)
a.plot(Ts, n, marker="s", ms=8, lw=2, color=C2, label="Newton (every pair once)", markeredgecolor="white", markeredgewidth=1.2)
a.plot(Ts, [o["newton_force_model_static_imbalance"] for o in out], ls=":", color=C2, lw=1.5, label="Newton, imbalance model")
a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xticks(Ts, [str(t) for t in Ts]); a.minorticks_off()
a.set_yticks([0.1, 0.2, 0.5, 1, 2], ["0.1", "0.2", "0.5", "1", "2"])
a.set_xlabel("OpenMP threads"); a.set_ylabel("median force time (s)")
a.set_title("(a) Force time: Newton wins only up to 2 threads", fontsize=11, loc="left"); a.legend(frameon=False, loc="lower left", fontsize=10)
r = [o["newton_over_direct_force"] for o in out]
b.axhline(1, ls="--", color="#8a8a85", lw=1.2)
b.plot(Ts, r, marker="s", ms=8, lw=2, color=C2, markeredgecolor="white", markeredgewidth=1.2)
for T, v in zip(Ts, r):
    b.annotate(f"{v:.2f}", (T, v), xytext=(6, 6), textcoords="offset points", color="#333")
b.text(1.05, 1.08, "Newton slower", color="#555", fontsize=10); b.text(1.05, 0.75, "Newton faster", color="#555", fontsize=10)
b.set_xscale("log", base=2); b.set_xticks(Ts, [str(t) for t in Ts]); b.minorticks_off()
b.set_xlabel("OpenMP threads"); b.set_ylabel("Newton force time / direct force time")
b.set_title("(b) Ratio: crossover between 2 and 4 threads", fontsize=11, loc="left")
for ax in (a, b):
    ax.grid(color="#e6e6e1"); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Newton's third law vs direct kernel, N=10000, 1 rank, exact sqrt (medians of 5 runs)", fontsize=12)
fig.tight_layout(); fig.savefig("newton_threads.svg")
for o in out:
    print(o["threads"], round(o["Kernel_direct_force_median"],4), round(o["Kernel_newton_force_median"],4), round(o["newton_force_model_static_imbalance"],4), round(o["newton_over_direct_force"],3), round(o["approx1_force_speedup"],2), round(o["approx2_force_speedup"],2))
