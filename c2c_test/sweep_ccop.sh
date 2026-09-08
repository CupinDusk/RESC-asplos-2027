#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# CCOP reuse/sharing sweep: A vs S1 across reuse degree R and consumer count.
# Deterministic cycle counts (host load irrelevant). Runs pairs in parallel.
# Usage: sweep_ccop.sh <tag>
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
TAG=${1:-sweep}
BASE=c2c_run/$TAG
NREG=32; RF=512; NC=2
launch () {  # arm mode noprobe reuse dir
  local arm=$1 mode=$2 noprobe=$3 rr=$4 d=$5
  mkdir -p $d; cp cfg_c2c.config $d/gpgpusim.config
  ( cd $d; nohup env RESC_C2C_MODE=$mode RESC_C2C_NOPROBE=$noprobe RESC_C2C_L2=242 \
      timeout 3600 ../../../resc_c2c $NREG $RF $rr $NC > run.log 2>&1 & )
}
for RR in 1 4 64; do
  launch A  0 0 $RR $BASE/R${RR}_A
  launch S1 1 0 $RR $BASE/R${RR}_S1
done
echo "SWEEP_LAUNCHED tag=$TAG (reuse 1,4,64 x {A,S1}; R=16 already in arms batch)"
