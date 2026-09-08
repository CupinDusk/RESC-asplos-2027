#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Experiment B: per-fan-in benefit(N) sweep WITH the symmetric injection charge.
# Usage: run_expB_sweep.sh <RESC_INJECT_BW> <avg_elems> [tag]
# N in {1,2,3,4,7} = cluster_size {2,3,4,5,8}. 3 arms. charge ON (frozen BW).
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cp cfg_h20.config gpgpusim.config
BW=${1:-15.34}; AE=${2:-2048}; TAG=${3:-bw${BW}_ae${AE}}
W=expB_sweep/$TAG; mkdir -p $W
echo "# ExpB benefit(N) sweep  RESC_INJECT_BW=$BW  avg_elems=$AE  cfg_h20 num_clusters=100 reuse=4"
echo "# N cluster global dsm resc_charged  benefit=global/resc  resc_beats_dsm  checksum_match  inject_events avg_queue"
declare -A CS=( [1]=2 [2]=3 [3]=4 [4]=5 [7]=8 )
for N in 1 2 3 4 7; do
  cs=${CS[$N]}
  # global (charge irrelevant; arm never calls read-share)
  env RESC_INJECT_CHARGE=0 ./global_ext cluster_size=$cs repeat=1 num_clusters=100 avg_elems=$AE burst_period=16 reuse_passes=4 > $W/global_N${N}.log 2>&1
  # dsm
  env RESC_INJECT_CHARGE=0 ./dsm_ext    cluster_size=$cs repeat=1 num_clusters=100 avg_elems=$AE burst_period=16 reuse_passes=4 > $W/dsm_N${N}.log 2>&1
  # resc WITH symmetric injection charge at frozen BW
  env RESC_INJECT_CHARGE=1 RESC_INJECT_BW=$BW ./resc_ext cluster_size=$cs repeat=1 num_clusters=100 avg_elems=$AE burst_period=16 reuse_passes=4 > $W/resc_N${N}.log 2>&1
  g=$(grep gpu_tot_sim_cycle $W/global_N${N}.log|tail -1|grep -oE "[0-9]+")
  d=$(grep gpu_tot_sim_cycle $W/dsm_N${N}.log|tail -1|grep -oE "[0-9]+")
  r=$(grep gpu_tot_sim_cycle $W/resc_N${N}.log|tail -1|grep -oE "[0-9]+")
  kg=$(grep -aoE "sum=[-0-9.eE+]+" $W/global_N${N}.log|head -1)
  kd=$(grep -aoE "sum=[-0-9.eE+]+" $W/dsm_N${N}.log|head -1)
  kr=$(grep -aoE "sum=[-0-9.eE+]+" $W/resc_N${N}.log|head -1)
  ev=$(grep RESC_INJECT_STATS $W/resc_N${N}.log|tail -1|grep -oE "events=[0-9]+"|cut -d= -f2)
  aq=$(grep RESC_INJECT_STATS $W/resc_N${N}.log|tail -1|grep -oE "avg_queue=[0-9.]+"|cut -d= -f2)
  ben=$(awk "BEGIN{if($r>0)printf \"%.4f\", $g/$r; else print \"NA\"}")
  bd=$(awk "BEGIN{print ($r<$d)?\"YES\":\"NO\"}")
  km=$([ "$kg" = "$kr" ] && [ "$kd" = "$kr" ] && echo "IDENTICAL($kr)" || echo "MISMATCH g=$kg d=$kd r=$kr")
  echo "$N $cs $g $d $r  $ben  $bd  $km  ev=$ev aq=$aq"
done
echo "EXPB_SWEEP_DONE $TAG"
