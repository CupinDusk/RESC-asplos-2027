#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Route 1 sim e2e: 3-arm x compute-scale sweep, PROVEN resc_e2e harness.
# fa_global/fa_dsm/fa_resc built with TILE_KB=32 COMPUTE_ITERS=4 (real flash-attn-shaped FMA,
# byte-identical across arms). Compute scale via cfg latency multiplier (cfg_cs*.config).
# Usage: run_fa_sweep.sh <cluster> <repeat> <ncl> <avg_elems> <reuse>
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
BIN=$PWD
CS=${1:-4}; REP=${2:-8}; NCL=${3:-8}; AVG=${4:-256}; REUSE=${5:-4}
TAG=fa_cs${CS}_r${REP}_n${NCL}_a${AVG}_ru${REUSE}
BASE=dec_run/sweep_$TAG; mkdir -p $BASE
for scale in 1_0 2_0 3_0 3_2 4_0 5_0; do
  for arm in global dsm resc; do
    d=$BASE/${scale}/${arm}; mkdir -p $d
    cp cfg_cs${scale}.config $d/gpgpusim.config
    ( cd $d; timeout 5000 $BIN/fa_$arm cluster_size=$CS repeat=$REP num_clusters=$NCL avg_elems=$AVG burst_period=16 reuse_passes=$REUSE > run.log 2>&1 ) &
  done
  wait
  echo "SCALE $scale DONE $(date +%H:%M:%S)"
done
echo "SWEEP_ALL_DONE"
echo "# scale arm cycles checksum"
for scale in 1_0 2_0 3_0 3_2 4_0 5_0; do
  for arm in global dsm resc; do
    d=$BASE/${scale}/${arm}
    c=$(grep "gpu_tot_sim_cycle =" $d/run.log 2>/dev/null|tail -1|grep -oE "[0-9]+")
    k=$(grep OUT_CHECKSUM $d/run.log 2>/dev/null|head -1|grep -oE "sum=[0-9.eE+-]+")
    echo "$scale $arm ${c:-NA} ${k:-NA}"
  done
done
