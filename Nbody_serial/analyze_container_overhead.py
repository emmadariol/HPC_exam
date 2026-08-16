#!/usr/bin/env python3
import csv
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    raise SystemExit("usage: analyze_container_overhead.py docker_overhead.csv [summary.csv]")

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else "container_overhead_summary.csv"

groups = defaultdict(list)
with open(src, newline="") as f:
    for row in csv.DictReader(f):
        key = (row.get("kind", "strong"), row["mode"], int(row["N"]), int(row["ranks"]), int(row["threads"]))
        groups[key].append(float(row["total"]))

configs = sorted({(kind, n, ranks, threads) for (kind, _, n, ranks, threads) in groups})
rows = []
for kind, n, ranks, threads in configs:
    native = groups.get((kind, "native", n, ranks, threads), [])
    docker = groups.get((kind, "docker", n, ranks, threads), []) or groups.get((kind, "container", n, ranks, threads), [])
    if not native or not docker:
        continue
    native_median = statistics.median(native)
    docker_median = statistics.median(docker)
    rows.append({
        "kind": kind,
        "N": n,
        "ranks": ranks,
        "threads": threads,
        "native_runs": len(native),
        "container_runs": len(docker),
        "native_median": native_median,
        "container_median": docker_median,
        "native_stdev": statistics.stdev(native) if len(native) > 1 else 0.0,
        "container_stdev": statistics.stdev(docker) if len(docker) > 1 else 0.0,
        "overhead_percent": 100.0 * (docker_median - native_median) / native_median,
    })

fields = [
    "kind", "N", "ranks", "threads", "native_runs", "container_runs",
    "native_median", "container_median", "native_stdev",
    "container_stdev", "overhead_percent",
]
with open(dst, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"wrote {dst}")
