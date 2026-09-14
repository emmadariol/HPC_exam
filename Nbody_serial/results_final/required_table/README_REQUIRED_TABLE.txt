Required container benchmark table
==================================

REQ_DIR=/u/dssc/emmadariol/HPC_exam/Nbody_serial/runs/required_table_20260912_225819
Generated at: Mon Sep 14 18:18:54 CEST 2026

Required experiments covered:
- Strong scaling native:    N=100000, 100 steps, P=1,2,4,8,16,32, 5 runs
- Strong scaling container: N=100000, 100 steps, P=1,2,4,8,16,32, 5 runs
- Weak scaling native:      N/P=10000, 100 steps, P=1,2,4,8,16, 5 runs
- Weak scaling container:   N/P=10000, 100 steps, P=1,2,4,8,16, 5 runs
- Launch overhead:          10 singularity exec launches
- OSU:                      latency + bandwidth, native vs container, 2 processes

Files:
results_final/required_table/README_REQUIRED_TABLE.txt
results_final/required_table/container_launch_overhead.csv
results_final/required_table/jobs_required.tsv
results_final/required_table/osu_microbench_bandwidth.svg
results_final/required_table/osu_microbench_latency.svg
results_final/required_table/osu_microbench_native_vs_container.csv
results_final/required_table/osu_microbench_summary.csv
results_final/required_table/required_container_scaling.csv
results_final/required_table/required_container_scaling_strong.svg
results_final/required_table/required_container_scaling_summary.csv
results_final/required_table/required_container_scaling_weak.svg

Jobs:
build	1626064
strong_native_P1	1626075
strong_container_P1	1626125
strong_native_P2	1626315
strong_native_P4	1626316
strong_native_P8	1626317
strong_container_P2	1626557
strong_container_P4	1626558
strong_container_P8	1626559
strong_container_P16	1626777
strong_container_P32	1626778
strong_native_P16	1637196
strong_native_P32	1637197
weak_native_P1	1637418
weak_native_P2	1637420
weak_native_P4	1637421
weak_native_P8	1637422
weak_native_P16	1637490
weak_container_P1	1641720
weak_container_P2	1641722
weak_container_P4	1641723
weak_container_P8	1642283
weak_container_P16	1642284
launch_overhead	
launch_overhead	1643782
osu	1643783
