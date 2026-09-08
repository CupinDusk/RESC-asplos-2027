#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Extract pipeline ablation results into a flat table.
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/selinv/pipe
for d in */; do
  d=${d%/}
  [ -f "$d/run.log" ] || continue
  cfg=$(echo $d | cut -d_ -f1)
  knob=$(echo $d | cut -d_ -f2 | tr -d k)
  lag=$(echo $d | cut -d_ -f3 | tr -d L)
  arm=$(echo $d | cut -d_ -f4)
  ec=$(cat $d/exit.code 2>/dev/null || echo NA)
  # the knob the SIMULATOR parsed, not what we wrote
  pk=$(grep -a "gpgpu_cluster_barrier_invalidate_l1" $d/run.log | head -1 | awk '{print $2}')
  cyc=$(grep -a "^gpu_tot_sim_cycle" $d/run.log | tail -1 | awk '{print $NF}')
  l1mr=$(grep -a "^\s*L1D_total_cache_miss_rate" $d/run.log | tail -1 | awk '{print $NF}')
  l2mr=$(grep -a "^\s*L2_total_cache_miss_rate" $d/run.log | tail -1 | awk '{print $NF}')
  dram=$(grep -a "^gpgpu_n_mem_read_global" $d/run.log | tail -1 | awk '{print $NF}')
  gs=$(grep -a "RESC_GEN_STATS" $d/run.log | tail -1)
  served=$(echo "$gs" | sed -n 's/.*readshare_served=\([0-9]*\).*/\1/p')
  expired=$(echo "$gs" | sed -n 's/.*readshare_expired=\([0-9]*\).*/\1/p')
  sp=$(grep -a "RESC_L1SPLIT_STATS" $d/run.log | tail -1)
  mh=$(echo "$sp" | sed -n 's/.*[^n]marked_rd_hit=\([0-9]*\).*/\1/p')
  mm=$(echo "$sp" | sed -n 's/.*[^n]marked_rd_miss=\([0-9]*\).*/\1/p')
  mhr=$(echo "$sp" | sed -n 's/.*[^n]marked_hitrate=\([0-9.]*\).*/\1/p')
  uh=$(echo "$sp" | sed -n 's/.*unmarked_rd_hit=\([0-9]*\).*/\1/p')
  um=$(echo "$sp" | sed -n 's/.*unmarked_rd_miss=\([0-9]*\).*/\1/p')
  uhr=$(echo "$sp" | sed -n 's/.*unmarked_hitrate=\([0-9.]*\).*/\1/p')
  si=$(grep -a "RESC_SELINV_STATS" $d/run.log | tail -1)
  wi=$(echo "$si" | sed -n 's/.*whole_invalidates=\([0-9]*\).*/\1/p')
  sv=$(echo "$si" | sed -n 's/.*selective_invalidates=\([0-9]*\).*/\1/p')
  mi=$(echo "$si" | sed -n 's/.*marked_invalidated=\([0-9]*\).*/\1/p')
  up=$(echo "$si" | sed -n 's/.*unmarked_preserved=\([0-9]*\).*/\1/p')
  orr=$(echo "$si" | sed -n 's/.*owner_retained=\([0-9]*\).*/\1/p')
  ck=$(grep -a "OUT_CHECKSUM\|checksum" $d/run.log | tail -1 | tr -d '\n' | sed 's/ /_/g' | cut -c1-60)
  echo "$cfg $knob ${pk:-NA} $lag $arm ${cyc:-NA} ${l1mr:-NA} ${l2mr:-NA} ${dram:-NA} ${served:-NA} ${expired:-NA} ${mh:-NA} ${mm:-NA} ${mhr:-NA} ${uh:-NA} ${um:-NA} ${uhr:-NA} ${wi:-NA} ${sv:-NA} ${mi:-NA} ${up:-NA} ${orr:-NA} $ec ${ck:-NA}"
done | sort -k1,1 -k2,2n -k4,4n -k5,5
