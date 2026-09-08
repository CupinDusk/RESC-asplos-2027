#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# RESC cross-cluster 3-arm sweep. Arms selected by RESC_C2C_MODE (0=A,1=B,2=C).
# Usage: run_c2c_arms.sh <nreg> <rfloats> <tag>
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
NREG=${1:-16}; RF=${2:-32}; TAG=${3:-run}
BASE=c2c_run/$TAG
for mode in 0 1 2; do
  case $mode in 0) arm=A;; 1) arm=B;; 2) arm=C;; esac
  d=$BASE/${arm}; mkdir -p $d
  cp cfg_c2c.config $d/gpgpusim.config
  ( cd $d; RESC_C2C_MODE=$mode RESC_C2C_K=4 RESC_C2C_L2=242 RESC_C2C_TIMEOUT=100 \
      timeout 900 ../../../resc_c2c $NREG $RF > run.log 2>&1 )
done
echo "# arm nreg rfloats cycles neighbor_hits l2_fallbacks pred_hits mispredicts checksum"
for mode in 0 1 2; do
  case $mode in 0) arm=A;; 1) arm=B;; 2) arm=C;; esac
  d=$BASE/${arm}
  cyc=$(grep "gpu_tot_sim_cycle =" $d/run.log | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
  read nh lf ph mp < <(grep "\[C2C_STATS\]" $d/run.log | tail -1 | sed -E "s/.*neighbor_hits=([0-9]+) l2_fallbacks=([0-9]+) pred_hits=([0-9]+) mispredicts=([0-9]+)/\1 \2 \3 \4/")
  ck=$(grep OUT_CHECKSUM $d/run.log | grep -oE "sum=[0-9]+" | head -1)
  echo "$arm $NREG $RF ${cyc:-NA} ${nh:-NA} ${lf:-NA} ${ph:-NA} ${mp:-NA} ${ck:-NA}"
done
echo C2C_ARMS_DONE
