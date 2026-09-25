"""Strong and weak scaling, native and in Singularity: N-body direct solver, 100 KDK steps,
blocking ring, 1 thread per rank, exact sqrt, energy checked at start and end, 5 seeds per point.
Native runs: strong_* and weak_* folders; container runs: ctr_* folders (same seeds).
Writes native_strong_summary.csv, native_weak_summary.csv, container_summary.csv and the figures."""
import csv, glob, statistics as st, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # svgplot.py lives in results_final
from svgplot import Chart, PALETTE, GREY


def load(pattern):
    groups = {}
    for f in sorted(glob.glob(pattern)):
        for r in csv.DictReader(open(f)):
            groups.setdefault((r["kind"], int(r["ranks"])), []).append(r)
    return groups


nat = load("[sw]*/scaling.csv")
ctr = load("ctr_*/scaling.csv")
med = lambda rows, k: st.median(float(r[k]) for r in rows)
sd = lambda rows, k: st.stdev(float(r[k]) for r in rows)
ok = lambda rows: all(r["status"] == "OK" for r in rows)
drift = lambda rows: max(float(r["max_rel_drift"]) for r in rows)


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


# ---- strong scaling (native)
Ps = sorted(p for k, p in nat if k == "strong")
b = nat[("strong", 1)]
T1, F1, E1 = med(b, "total"), med(b, "force"), med(b, "energy")
strong = []
for P in Ps:
    g = nat[("strong", P)]
    t, f, e = med(g, "total"), med(g, "force"), med(g, "energy")
    strong.append({"ranks": P, "N": 100000, "runs": len(g), "all_ok": ok(g),
                   "total_median": t, "total_stdev": sd(g, "total"),
                   "speedup": T1 / t, "efficiency": T1 / t / P,
                   "force_median": f, "efficiency_force": F1 / f / P,
                   "comm_wait_median": med(g, "comm_wait"), "comm_wait_percent": 100 * med(g, "comm_wait") / t,
                   "energy_median": e, "efficiency_energy": E1 / e / P, "energy_percent": 100 * e / t,
                   "io_median": med(g, "io"), "gpairs_per_rank": med(g, "gpairs") / P,
                   "max_rel_drift": drift(g)})
write("native_strong_summary.csv", strong)

# ---- weak scaling (native): ideal time grows like the number of pairs per rank
Pw = sorted(p for k, p in nat if k == "weak")
w1 = nat[("weak", 1)]
W1 = med(w1, "total")
n1 = 10000
weak = []
for P in Pw:
    g = nat[("weak", P)]
    N = n1 * P
    R = N * (N - 1) / (n1 * (n1 - 1))  # pairs ratio vs one rank
    t = med(g, "total")
    weak.append({"ranks": P, "N": N, "runs": len(g), "all_ok": ok(g),
                 "total_median": t, "total_stdev": sd(g, "total"),
                 "ideal_time": W1 * R / P, "time_over_ideal": t / (W1 * R / P),
                 "work_speedup": R * W1 / t, "work_efficiency": R * W1 / t / P,
                 "force_median": med(g, "force"), "comm_wait_percent": 100 * med(g, "comm_wait") / t,
                 "energy_percent": 100 * med(g, "energy") / t, "max_rel_drift": drift(g)})
write("native_weak_summary.csv", weak)

# ---- native vs container
cont = []
for (k, P) in sorted(ctr, key=lambda x: (x[0], x[1])):
    gn, gc = nat[(k, P)], ctr[(k, P)]
    tn, tc = med(gn, "total"), med(gc, "total")
    cont.append({"kind": k, "ranks": P, "N": int(gn[0]["N"]), "runs_native": len(gn), "runs_container": len(gc),
                 "all_ok": ok(gn) and ok(gc),
                 "native_median": tn, "native_stdev": sd(gn, "total"),
                 "container_median": tc, "container_stdev": sd(gc, "total"),
                 "overhead_percent": 100 * (tc / tn - 1),
                 "combined_spread_percent": 100 * (sd(gn, "total") ** 2 + sd(gc, "total") ** 2) ** 0.5 / tn,
                 "same_energy_drift": all(abs(float(a["max_rel_drift"]) - float(c["max_rel_drift"])) == 0
                                          for a, c in zip(sorted(gn, key=lambda r: r["repeat"]), sorted(gc, key=lambda r: r["repeat"])))})
write("container_summary.csv", cont)

# ---- figures (same look as the other report figures)
xl, xi = [str(p) for p in Ps], list(range(len(Ps)))
c = Chart("Strong scaling N=100000, 100 steps: run time", "MPI ranks P (one thread each)", "median total time (s)",
          xl, 0, T1 * 1.05, yfmt="{:.0f}", legend="tr")
c.line(xi, [T1 / p for p in Ps], GREY, dash="6,4", r=0, label="ideal T(1)/P")
c.line(xi, [s["total_median"] for s in strong], PALETTE[0], label="measured")
c.save("strong_runtime.svg")

c = Chart("Strong scaling N=100000, 100 steps: speedup", "MPI ranks P (one thread each)", "speedup T(1)/T(P)",
          xl, 0, 32, yticks=[0, 8, 16, 24, 32])
c.line(xi, Ps, GREY, dash="6,4", r=0, label="ideal")
c.line(xi, [s["speedup"] for s in strong], PALETTE[0], label="total time")
c.line(xi, [F1 / s["force_median"] for s in strong], PALETTE[1], label="force phase")
c.text(xi[-1], strong[-1]["speedup"], f"{strong[-1]['speedup']:.1f}", dx=30, dy=40, anchor="end")
c.save("strong_speedup.svg")

c = Chart("Strong scaling N=100000, 100 steps: efficiency by phase", "MPI ranks P (one thread each)",
          "parallel efficiency S(P)/P", xl, 0, 1.1, yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], legend="bl")
c.hline(1.0)
c.line(xi, [s["efficiency"] for s in strong], PALETTE[0], label="total time")
c.line(xi, [s["efficiency_force"] for s in strong], PALETTE[1], label="force phase")
c.line(xi, [s["efficiency_energy"] for s in strong], PALETTE[2], label="energy check")
c.legend("ideal", GREY)
c.save("strong_efficiency.svg")

xw, xwi = [str(p) for p in Pw], list(range(len(Pw)))
c = Chart("Weak scaling, 10000 particles per rank: run time", "MPI ranks P (N = 10000 x P)", "median total time (s)",
          xw, 0, max(s["total_median"] for s in weak) * 1.1, yfmt="{:.0f}")
c.line(xwi, [s["ideal_time"] for s in weak], GREY, dash="6,4", r=0, label="ideal: time grows like P")
c.line(xwi, [s["total_median"] for s in weak], PALETTE[0], label="measured")
c.save("weak_time.svg")

c = Chart("Weak scaling: work-normalised efficiency", "MPI ranks P (N = 10000 x P)", "efficiency",
          xw, 0, 1.1, yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], legend="bl")
c.hline(1.0)
c.line(xwi, [s["work_efficiency"] for s in weak], PALETTE[0], label="measured")
c.legend("ideal", GREY)
c.save("weak_efficiency.svg")

labels = [f"{'strong' if o['kind'] == 'strong' else 'weak'} P={o['ranks']}" for o in cont]
lim = max(abs(o["overhead_percent"]) + o["combined_spread_percent"] for o in cont) * 1.3
c = Chart("Container vs native: change of median total time", "configuration",
          "container vs native (%)", labels, -lim, lim, pad=True, yfmt="{:+.1f}%", legend="tr")
c.hline(0.0, dash=None)
for i, o in enumerate(cont):
    col = PALETTE[0] if o["kind"] == "strong" else PALETTE[1]
    c.errorbar(i, o["overhead_percent"], o["combined_spread_percent"], col)
    c.text(i, o["overhead_percent"], f"{o['overhead_percent']:+.2f}%", dx=8, dy=-10, size=11)
c.legend("strong scaling, N = 100000", PALETTE[0])
c.legend("weak scaling, 10000 per rank", PALETTE[1])
c.save("container_overhead.svg")

for s in strong: print("S", s["ranks"], round(s["total_median"], 2), round(s["total_stdev"], 2), round(s["speedup"], 2), round(100*s["efficiency"], 1), round(100*s["efficiency_force"], 1), round(100*s["efficiency_energy"], 1), round(s["comm_wait_percent"], 2), round(s["energy_percent"], 2), round(s["gpairs_per_rank"], 4), f"{s['max_rel_drift']:.2e}")
for s in weak: print("W", s["ranks"], s["N"], round(s["total_median"], 2), round(s["total_stdev"], 2), round(s["ideal_time"], 1), round(s["time_over_ideal"], 3), round(s["work_speedup"], 2), round(100*s["work_efficiency"], 1), round(s["comm_wait_percent"], 2), round(s["energy_percent"], 2), f"{s['max_rel_drift']:.2e}")
for o in cont: print("C", o["kind"], o["ranks"], o["N"], round(o["native_median"], 2), round(o["native_stdev"], 2), round(o["container_median"], 2), round(o["container_stdev"], 2), round(o["overhead_percent"], 2), round(o["combined_spread_percent"], 2), o["same_energy_drift"])
