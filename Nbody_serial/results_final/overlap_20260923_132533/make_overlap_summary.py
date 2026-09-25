"""Summarise the matched sendrecv/overlap campaign and draw the report figure."""
import csv, glob, statistics as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # svgplot.py lives in results_final
from svgplot import Chart, PALETTE, GREY

rows = {}
for f in sorted(glob.glob("P*_*/scaling.csv")):
    for r in csv.DictReader(open(f)):
        rows.setdefault((int(r["ranks"]), r["comm"]), []).append(r)

def col(P, c, k):
    return [float(x[k]) for x in rows[(P, c)]]

Ps = sorted({p for p, _ in rows})
out = []
for P in Ps:
    rec = {"ranks": P, "N": 100000, "nsteps": 20, "threads": 1}
    for c in ("sendrecv", "overlap"):
        for k in ("total", "force", "comm_wait", "energy", "io"):
            v = col(P, c, k)
            rec[f"{c}_{k}_median"] = st.median(v)
            rec[f"{c}_{k}_stdev"] = st.stdev(v)
        rec[f"{c}_runs"] = len(rows[(P, c)])
        rec[f"{c}_all_ok"] = all(x["status"] == "OK" for x in rows[(P, c)])
        rec[f"{c}_max_rel_drift"] = max(float(x["max_rel_drift"]) for x in rows[(P, c)])
    ws, wo = rec["sendrecv_comm_wait_median"], rec["overlap_comm_wait_median"]
    rec["hidden_wait_s"] = ws - wo
    rec["overlap_fraction"] = (ws - wo) / ws
    rec["total_ratio_overlap_over_sendrecv"] = rec["overlap_total_median"] / rec["sendrecv_total_median"]
    rec["force_ratio_overlap_over_sendrecv"] = rec["overlap_force_median"] / rec["sendrecv_force_median"]
    rec["sendrecv_wait_fraction_of_total"] = ws / rec["sendrecv_total_median"]
    rec["wait_per_ring_phase_ms"] = 1e3 * ws / (21 * P)  # P exchanges per force evaluation
    rec["force_per_ring_phase_ms"] = 1e3 * rec["sendrecv_force_median"] / (21 * P)
    out.append(rec)

with open("overlap_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader(); w.writerows(out)

xl = [f"P={P}" for P in Ps]
ymax = max(max(col(P, c_, "comm_wait")) for P in Ps for c_ in ("sendrecv", "overlap")) * 1.15
c = Chart("Exposed ring wait per run, N=100000, 20 steps, T=1", "MPI ranks", "exposed comm_wait per run (s)",
          xl, 0, ymax, pad=True, yfmt="{:.2f}", legend="tr")
for c_, dx, color in (("sendrecv", -0.13, PALETTE[0]), ("overlap", 0.13, PALETTE[1])):
    for i, P in enumerate(Ps):
        v = col(P, c_, "comm_wait")
        c.points([i + dx] * len(v), v, color, 4)
        m = st.median(v)
        c.seg(i + dx - 0.09, m, i + dx + 0.09, m)
    c.legend(c_ + " (bar = median of 5 runs)", color)
c.save("ring_overlap_wait.svg")
diff = [100 * (r["total_ratio_overlap_over_sendrecv"] - 1) for r in out]
noise = [100 * (r["sendrecv_total_stdev"]**2 + r["overlap_total_stdev"]**2) ** 0.5 / r["sendrecv_total_median"] for r in out]
lim = max(abs(d) + e for d, e in zip(diff, noise)) * 1.3
c = Chart("Total time: overlap vs sendrecv (error bar = combined spread)", "MPI ranks",
          "change of median total time (%)", xl, -lim, lim, pad=True, yfmt="{:+.1f}%")
c.hline(0.0, dash=None)
for i, (d, e) in enumerate(zip(diff, noise)):
    c.errorbar(i, d, e, PALETTE[1])
    c.text(i, d, f"{d:+.2f}%", dx=14, dy=4)
c.save("ring_overlap_total.svg")
