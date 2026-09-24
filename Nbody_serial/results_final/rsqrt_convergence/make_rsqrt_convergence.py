"""Time-step convergence of the energy drift for exact/approx1/approx2 reciprocal sqrt.
Fixed physical time 0.01; dt = 1e-4, 5e-5, 2.5e-5 (100, 200, 400 KDK steps);
N=10000 Plummer, P=1, T=8, energy every step; three seeds per point.
Reads ../rsqrt_conv_<method>_n<steps>_20260923_170127/scaling.csv
"""
import csv, glob, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = {}
for f in glob.glob("../rsqrt_conv_*_20260923_170127/scaling.csv"):
    for r in csv.DictReader(open(f)):
        data.setdefault((r["rsqrt"], int(r["nsteps"])), []).append(r)
methods = ["exact", "approx1", "approx2"]
steps = [100, 200, 400]
out = []
for m in methods:
    for n in steps:
        R = sorted(data[(m, n)], key=lambda x: int(x["repeat"]))
        d = [float(x["max_rel_drift"]) for x in R]
        dex = [float(x["max_rel_drift"]) for x in sorted(data[("exact", n)], key=lambda x: int(x["repeat"]))]
        out.append({"rsqrt": m, "nsteps": n, "dt": 0.01 / n, "runs": len(d),
                    "all_ok": all(x["status"] == "OK" for x in R),
                    "drift_seed1": d[0], "drift_seed2": d[1], "drift_seed3": d[2],
                    "drift_max": max(d),
                    "max_abs_diff_vs_exact": max(abs(a - b) for a, b in zip(d, dex)),
                    "median_force_s": st.median(float(x["force"]) for x in R)})
for m in methods:
    rows = [o for o in out if o["rsqrt"] == m]
    for a, b in zip(rows, rows[1:]):
        b["ratio_prev_dt_over_this"] = a["drift_max"] / b["drift_max"]
keys = list(out[0]) + ["ratio_prev_dt_over_this"]
with open("rsqrt_convergence_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)

C = {"exact": "#2a78d6", "approx1": "#eb6834", "approx2": "#1baf7a"}
MK = {"exact": "o", "approx1": "s", "approx2": "^"}
SZ = {"exact": 11, "approx1": 7, "approx2": 5}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11,
                     "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#333",
                     "xtick.color": "#555", "ytick.color": "#555"})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.8))
dts = [0.01 / n for n in steps]
lab = {1e-4: "1e-4", 5e-5: "5e-5", 2.5e-5: "2.5e-5"}
ref0 = [o for o in out if o["rsqrt"] == "exact"][0]["drift_max"]
ax.plot(dts, [ref0 * (dt / dts[0]) ** 2 for dt in dts], ls="--", color="#8a8a85", lw=1.2, label="slope 2 (ideal 2nd order)")
for m in methods:
    y = [o["drift_max"] for o in out if o["rsqrt"] == m]
    ax.plot(dts, y, marker=MK[m], ms=SZ[m], lw=2, color=C[m], label=m, markeredgecolor="white", markeredgewidth=1.2)
exact_y = [o["drift_max"] for o in out if o["rsqrt"] == "exact"]
for dt, y in zip(dts, exact_y):
    ax.annotate(f"{y:.2e}", (dt, y), xytext=(8, -12), textcoords="offset points", color="#333", fontsize=10)
ax.set_title("(a) Energy drift vs dt: all three curves coincide", fontsize=11, loc="left")
ax.legend(frameon=False, loc="upper left")
bx.plot(dts, exact_y, marker="o", ms=8, lw=2, color=C["exact"], label="exact drift (integration error)", markeredgecolor="white", markeredgewidth=1.2)
d1 = [o["max_abs_diff_vs_exact"] for o in out if o["rsqrt"] == "approx1"]
bx.plot(dts, d1, marker="s", ms=8, lw=2, color=C["approx1"], label="|approx1 - exact| drift difference", markeredgecolor="white", markeredgewidth=1.2)
for dt, y in zip(dts, d1):
    bx.annotate(f"{y:.1e}", (dt, y), xytext=(8, 8), textcoords="offset points", color="#333", fontsize=10)
bx.text(0.02, 0.04, "approx2: difference from exact = 0 at all dt\n(identical to all printed digits; not drawn on log axis)", transform=bx.transAxes, fontsize=10, color="#333")
bx.set_title("(b) Approximation error stays far below integration error", fontsize=11, loc="left")
bx.legend(frameon=False, loc="center right")
for a_ in (ax, bx):
    a_.set_xscale("log"); a_.set_yscale("log")
    a_.set_xticks(dts, [lab[d] for d in dts]); a_.minorticks_off()
    a_.set_xlabel("time step dt (fixed physical time 0.01)")
    a_.grid(color="#e6e6e1", which="major"); a_.set_axisbelow(True)
    a_.spines[["top", "right"]].set_visible(False)
ax.set_ylabel("max relative energy drift (3 seeds)")
bx.set_ylabel("relative energy drift")
bx.set_ylim(1e-13, 1e-6)
fig.suptitle("Time-step convergence with exact and approximate rsqrt: N=10000 Plummer, P=1, T=8, energy every step", fontsize=12)
fig.tight_layout(); fig.savefig("rsqrt_convergence.svg")
