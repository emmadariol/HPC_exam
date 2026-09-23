"""Derive native N=100000 strong-scaling statistics and figures from raw runs."""
from pathlib import Path
import csv
import statistics
from analyze import plot_xy, write_csv


def build():
    root = Path(__file__).resolve().parent / 'results_final' / 'required_table'
    with (root / 'required_container_scaling.csv').open() as stream:
        raw = [r for r in csv.DictReader(stream)
               if r['kind'] == 'strong' and r['mode'] == 'native']
    baseline = statistics.median(float(r['total']) for r in raw if r['ranks'] == '1')
    rows = []
    for ranks in sorted({int(r['ranks']) for r in raw}):
        samples = [r for r in raw if int(r['ranks']) == ranks]
        assert len(samples) == 5
        assert all(r['status'] == 'OK' and r['N'] == '100000' and r['threads'] == '1'
                   for r in samples)
        values = [float(r['total']) for r in samples]
        median = statistics.median(values)
        rows.append(dict(N=100000, ranks=ranks, threads=1, resources=ranks,
                         runs=len(samples), total_median=median,
                         total_stdev=statistics.stdev(values), speedup=baseline/median,
                         efficiency=baseline/median/ranks,
                         max_rel_drift=max(float(r['max_rel_drift']) for r in samples)))
    write_csv(root / 'required_native_strong_summary.csv', list(rows[0]), rows)
    for field, suffix, title, ylabel, ideal in [
        ('total_median', 'runtime', 'Native strong scaling: N=100000', 'median time (s)', 'runtime'),
        ('speedup', 'speedup', 'Native strong scaling: N=100000', 'speedup T(1)/T(P)', 'speedup'),
        ('efficiency', 'efficiency', 'Native strong scaling: N=100000', 'efficiency S(P)/P', 'efficiency'),
    ]:
        plot_xy(rows, 'resources', field, title, ylabel,
                str(root / f'required_native_strong_{suffix}.svg'), ideal)
    return rows


if __name__ == '__main__':
    for row in build():
        print(row)
