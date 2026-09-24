"""Summarise the matched sendrecv/overlap campaign and draw the report figure."""
import csv, glob, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    rec["wait_per_ring_phase_ms"] = 1e3 * ws / (21 * (P - 1))
    rec["force_per_ring_phase_ms"] = 1e3 * rec["sendrecv_force_median"] / (21 * P)
    out.append(rec)

with open("overlap_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader(); w.writerows(out)

C = {"sendrecv": "#2a78d6", "overlap": "#eb6834"}
MK = {"sendrecv": "o", "overlap": "s"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11,
                     "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#333",
                     "xtick.color": "#555", "ytick.color": "#555"})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
xs = {P: i for i, P in enumerate(Ps)}
for c, dx in (("sendrecv", -0.13), ("overlap", 0.13)):
    for P in Ps:
        v = col(P, c, "comm_wait")
        a.scatter([xs[P] + dx] * len(v), v, s=42, marker=MK[c], color=C[c],
                  edgecolor="white", linewidth=1.5, zorder=3,
                  label=c if P == Ps[0] else None)
        a.hlines(st.median(v), xs[P] + dx - 0.09, xs[P] + dx + 0.09, color="#333", lw=2, zorder=4)
a.set_xticks(range(len(Ps)), [f"P={P}" for P in Ps])
a.set_ylabel("exposed comm_wait per run (s)")
a.set_title("(a) Exposed ring wait, 5 runs per mode (bar = median)", fontsize=11, loc="left")
a.legend(frameon=False, loc="upper right", ncol=2)
a.grid(axis="y", color="#e6e6e1"); a.set_axisbelow(True)
a.set_ylim(0, None)
diff = [100 * (r["total_ratio_overlap_over_sendrecv"] - 1) for r in out]
noise = [100 * (r["sendrecv_total_stdev"]**2 + r["overlap_total_stdev"]**2) ** 0.5 / r["sendrecv_total_median"] for r in out]
b.axhline(0, color="#8a8a85", lw=1)
b.errorbar(range(len(Ps)), diff, yerr=noise, fmt="o", color=C["overlap"], ecolor="#8a8a85",
           elinewidth=1.5, capsize=5, markersize=8, markeredgecolor="white", zorder=3)
for i, d in enumerate(diff):
    b.annotate(f"{d:+.2f}%", (i, d), xytext=(12, -4), textcoords="offset points", color="#333")
b.set_xticks(range(len(Ps)), [f"P={P}" for P in Ps])
b.set_xlim(-0.5, len(Ps) - 0.5)
b.set_ylabel("overlap vs sendrecv, median total (%)")
b.set_title("(b) Total-time change; bars = combined sample s", fontsize=11, loc="left")
b.grid(axis="y", color="#e6e6e1"); b.set_axisbelow(True)
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Blocking vs non-blocking ring: N=100000, 20 steps, T=1, one GENOA node", fontsize=12)
fig.tight_layout()
fig.savefig("ring_overlap.svg")
