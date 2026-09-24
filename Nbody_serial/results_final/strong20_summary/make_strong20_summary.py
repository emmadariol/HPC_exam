"""Strong scaling N=100000, 20 steps, blocking ring, T=1, same executable for all P.
P=1,4,16,64 from ../strong20_*_20260923_165910 ; P=2,8,32 from the sendrecv runs of
../overlap_20260923_132533/P*_sendrecv.
"""
import csv, glob, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = {}
files = glob.glob("../strong20_*_20260923_165910/scaling.csv") + glob.glob("../overlap_20260923_132533/P*_sendrecv/scaling.csv")
for f in files:
    for r in csv.DictReader(open(f)):
        if r["comm"] != "sendrecv":
            continue
        rows.setdefault(int(r["ranks"]), []).append(r)
Ps = sorted(rows)
med = lambda P, k: st.median(float(x[k]) for x in rows[P])
sd = lambda P, k: st.stdev(float(x[k]) for x in rows[P])
T1, F1, E1 = med(1, "total"), med(1, "force"), med(1, "energy")
out = []
for P in Ps:
    t, f, e = med(P, "total"), med(P, "force"), med(P, "energy")
    out.append({"ranks": P, "runs": len(rows[P]), "all_ok": all(x["status"] == "OK" for x in rows[P]),
                "total_median": t, "total_stdev": sd(P, "total"),
                "force_median": f, "comm_wait_median": med(P, "comm_wait"), "energy_median": e,
                "io_median": med(P, "io"), "gpairs_median": med(P, "gpairs"),
                "gpairs_per_rank": med(P, "gpairs") / P,
                "speedup_total": T1 / t, "efficiency_total": T1 / t / P,
                "speedup_force": F1 / f, "efficiency_force": F1 / f / P,
                "efficiency_energy": E1 / e / P,
                "max_rel_drift": max(float(x["max_rel_drift"]) for x in rows[P])})
with open("strong20_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": "#8a8a85",
                     "axes.labelcolor": "#333", "xtick.color": "#555", "ytick.color": "#555"})
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))
a.plot(Ps, Ps, ls="--", color="#8a8a85", lw=1.2, label="ideal")
a.plot(Ps, [o["speedup_total"] for o in out], marker="o", ms=8, lw=2, color=C1, label="total time", markeredgecolor="white", markeredgewidth=1.2)
a.plot(Ps, [o["speedup_force"] for o in out], marker="s", ms=6, lw=2, color=C2, label="force phase", markeredgecolor="white", markeredgewidth=1.2)
a.annotate(f"{out[-1]['speedup_total']:.1f}", (Ps[-1], out[-1]["speedup_total"]), xytext=(-34, -4), textcoords="offset points", color="#333")
a.set_xscale("log", base=2); a.set_yscale("log", base=2)
a.set_xticks(Ps, [str(p) for p in Ps]); a.set_yticks(Ps, [str(p) for p in Ps]); a.minorticks_off()
a.set_xlabel("MPI ranks P (one thread each)"); a.set_ylabel("speedup T(1)/T(P)")
a.set_title("(a) Speedup", fontsize=11, loc="left"); a.legend(frameon=False, loc="upper left")
b.axhline(1, ls="--", color="#8a8a85", lw=1.2)
b.plot(Ps, [o["efficiency_total"] for o in out], marker="o", ms=8, lw=2, color=C1, label="total time", markeredgecolor="white", markeredgewidth=1.2)
b.plot(Ps, [o["efficiency_force"] for o in out], marker="s", ms=6, lw=2, color=C2, label="force phase", markeredgecolor="white", markeredgewidth=1.2)
b.plot(Ps, [o["efficiency_energy"] for o in out], marker="^", ms=7, lw=2, color=C3, label="energy check", markeredgecolor="white", markeredgewidth=1.2)
b.set_xscale("log", base=2); b.set_xticks(Ps, [str(p) for p in Ps]); b.minorticks_off()
b.set_ylim(0.4, 1.08); b.set_xlabel("MPI ranks P (one thread each)"); b.set_ylabel("parallel efficiency S(P)/P")
b.set_title("(b) Efficiency by phase", fontsize=11, loc="left"); b.legend(frameon=False, loc="lower left")
for ax in (a, b):
    ax.grid(color="#e6e6e1"); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Strong scaling, N=100000, 20 steps, same executable for all P (medians of 5 runs)", fontsize=12)
fig.tight_layout(); fig.savefig("strong20_scaling.svg")
for o in out:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()})
