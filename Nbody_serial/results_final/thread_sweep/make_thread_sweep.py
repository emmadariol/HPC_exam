"""Kernel ablation vs OpenMP threads: N=10000, 5 steps, P=1, T=1,2,4,8,16.
T=1 from ../ablation_complete_20260922_083515, T>1 from ../ablation_T*_20260923_165910."""
import csv, glob, statistics as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # svgplot.py lives in results_final
from svgplot import Chart, PALETTE, GREY

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

xl = [str(t) for t in Ts]
xi = list(range(len(Ts)))
d = [o["Kernel_direct_force_median"] for o in out]
n = [o["Kernel_newton_force_median"] for o in out]
c = Chart("Newton vs direct kernel: force time, N=10000, 1 rank", "OpenMP threads", "median force time (s)",
          xl, 0, max(d + n) * 1.05, legend="tr")
c.line(xi, [d[0] / T for T in Ts], GREY, dash="6,4", r=0, label="direct, ideal")
c.line(xi, d, PALETTE[0], label="direct (every pair twice)")
c.line(xi, n, PALETTE[1], label="Newton (every pair once)")
c.line(xi, [o["newton_force_model_static_imbalance"] for o in out], PALETTE[1], dash="2,4", r=0, label="Newton, imbalance model")
c.save("newton_threads_time.svg")
r = [o["newton_over_direct_force"] for o in out]
c = Chart("Newton force time / direct force time", "OpenMP threads", "ratio Newton / direct",
          xl, 0, 2.0, yticks=[0, 0.5, 1.0, 1.5, 2.0])
c.hline(1.0)
c.line(xi, r, PALETTE[1])
for i, v in zip(xi, r):
    c.text(i, v, f"{v:.2f}", dx=8, dy=-8 if v > 1 else 20)
c.text(0, 1.0, "Newton slower", dx=10, dy=-10, color=GREY, size=13)
c.text(0, 1.0, "Newton faster", dx=10, dy=22, color=GREY, size=13)
c.save("newton_threads_ratio.svg")
for o in out:
    print(o["threads"], round(o["Kernel_direct_force_median"],4), round(o["Kernel_newton_force_median"],4), round(o["newton_force_model_static_imbalance"],4), round(o["newton_over_direct_force"],3), round(o["approx1_force_speedup"],2), round(o["approx2_force_speedup"],2))
