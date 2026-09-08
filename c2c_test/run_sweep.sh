#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Route 1 sim e2e: 3-arm x compute-scale sweep on the real decode kernel.
# Usage: run_sweep.sh <cluster> <repeat> <ncl>
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
BIN=$PWD
CS=${1:-4}; REP=${2:-16}; NCL=${3:-8}
TAG=cs${CS}_r${REP}_n${NCL}
BASE=dec_run/sweep_$TAG; mkdir -p $BASE
for scale in 1_0 2_0 3_0 3_2 4_0 5_0; do
  for arm in global dsm resc; do
    d=$BASE/${scale}/${arm}; mkdir -p $d
    cp cfg_cs${scale}.config $d/gpgpusim.config
    ( cd $d; timeout 3000 $BIN/dec_$arm $CS $REP $NCL > run.log 2>&1 ) &
  done
  wait   # finish all 3 arms of this scale before next (bounds concurrency to 3)
  echo "SCALE $scale DONE $(date +%H:%M:%S)"
done
echo "SWEEP_ALL_DONE"
# summary
echo "# scale arm cycles checksum"
for scale in 1_0 2_0 3_0 3_2 4_0 5_0; do
  for arm in global dsm resc; do
    d=$BASE/${scale}/${arm}
    c=$(grep "gpu_tot_sim_cycle =" $d/run.log 2>/dev/null|tail -1|grep -oE "[0-9]+")
    k=$(grep OUT_CHECKSUM $d/run.log 2>/dev/null|head -1|grep -oE "sum=[0-9.eE+-]+")
    echo "$scale $arm ${c:-NA} ${k:-NA}"
  done
done
