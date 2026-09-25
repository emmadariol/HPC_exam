"""Time-step convergence of the energy drift for exact/approx1/approx2 reciprocal sqrt.
Fixed physical time 0.01; dt = 1e-4, 5e-5, 2.5e-5 (100, 200, 400 KDK steps);
N=10000 Plummer, P=1, T=8, energy every step; three seeds per point.
Reads ../rsqrt_conv_<method>_n<steps>_20260923_170127/scaling.csv
"""
import csv, glob, statistics as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # svgplot.py lives in results_final
from svgplot import Chart, PALETTE, GREY

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

order = [400, 200, 100]  # smallest dt on the left
xl = ["2.5e-5", "5e-5", "1e-4"]
xi = [0, 1, 2]
get = lambda m, k: [next(o[k] for o in out if o["rsqrt"] == m and o["nsteps"] == s) for s in order]
exact_y = get("exact", "drift_max")
c = Chart("Energy drift vs time step: exact and approximate rsqrt", "time step dt (fixed physical time 0.01)",
          "max relative energy drift (3 seeds)", xl, 1e-8, 1e-6, ylog=True)
c.line(xi, [exact_y[2] * (0.01 / s / 1e-4) ** 2 for s in order], GREY, dash="6,4", r=0, label="slope 2 (ideal second order)")
for m, col, rad in (("exact", PALETTE[0], 8), ("approx1", PALETTE[1], 5.5), ("approx2", PALETTE[2], 3)):
    c.line(xi, get(m, "drift_max"), col, r=rad, label=m)
for i, y in zip(xi, exact_y):
    c.text(i, y, f"{y:.2e}", dx=25 if i == 2 else 10, dy=30 if i == 2 else 18, anchor="end" if i == 2 else "start")
c.save("rsqrt_convergence_drift.svg")
d1 = get("approx1", "max_abs_diff_vs_exact")
c = Chart("Approximation error vs integration error", "time step dt (fixed physical time 0.01)",
          "relative energy drift", xl, 1e-13, 1e-6, ylog=True)
c.line(xi, exact_y, PALETTE[0], label="exact drift (integration error)")
c.line(xi, d1, PALETTE[1], label="|approx1 - exact| drift difference")
for i, y in zip(xi, d1):
    c.text(i, y, f"{y:.1e}", dx=-8 if i == 2 else 8, dy=-10, anchor="end" if i == 2 else "start")
c.note(c.l + 10, c.Y(3e-10), "approx2: difference from exact = 0 at every dt", color=PALETTE[2])
c.note(c.l + 10, c.Y(3e-10) + 18, "(identical to all printed digits, cannot be drawn on a log axis)", color=GREY, size=12)
c.save("rsqrt_convergence_error.svg")
