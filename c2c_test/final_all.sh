#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
source ~/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cp cfg_h20.config gpgpusim.config

# wait for any in-flight arm runner
while pgrep -f "run_mw_arms.sh" >/dev/null; do sleep 5; done

# Sanity gate on final lib
SAN=$(./simxl_c8 256 2>/dev/null | grep -a "cyc/read" | tail -1)

TO=250
run4() {  # $1 bin $2 base $3 nc $4 rr $5 label $6 grep
  local BIN="$1" BASE="$2" NC="$3" RR="$4" LABEL="$5" GREP="$6"
  echo "################################################################"
  echo "### KERNEL: $LABEL   (num_clusters=$NC rounds/iters=$RR)"
  echo "################################################################"
  echo "--- arm1 CORRECT-GLOBAL (no RESC) ---"
  timeout $TO ./$BASE $NC $RR 2>/dev/null | grep -iE "$GREP|RESC_MW_STATS|gpu_tot_sim_cycle ="
  echo "--- arm2 RESC-MULTI-WRITER (RESC_MW=1) ---"
  RESC_MW=1 timeout $TO ./$BIN $NC $RR 2>/dev/null | grep -iE "$GREP|RESC_MW_STATS|gpu_tot_sim_cycle ="
  echo "--- arm3 RESC-OLD-ESCAPE (RESC_MW=0) ---"
  RESC_MW=0 timeout $TO ./$BIN $NC $RR 2>/dev/null | grep -iE "$GREP|RESC_MW_STATS|gpu_tot_sim_cycle ="
  echo "--- arm4 RESC-WHOLE-LINE-LWW (RESC_MW=1 RESC_MW_LWW=1) ---"
  RESC_MW=1 RESC_MW_LWW=1 timeout $TO ./$BIN $NC $RR 2>/dev/null | grep -iE "$GREP|RESC_MW_STATS|gpu_tot_sim_cycle ="
}

{
  echo "SANITY GATE (final lib): $SAN"
  echo ""
  run4 mw_reduce_fs     mw_reduce_fs_base     8 5  "PRIMARY-A false-sharing reduction (FORM=1)"   "MW_REDUCE"
  echo ""
  run4 mw_reduce_atomic mw_reduce_atomic_base 8 1  "PRIMARY-B atomic reduction (FORM=0, rounds=1 due to pre-existing coalescer bug)"  "MW_REDUCE"
  echo ""
  run4 mw_stencil       mw_stencil_base       8 10 "SECONDARY stencil false-sharing (Jacobi)"     "MW_STENCIL"
} > /tmp/final_raw.txt 2>&1
echo FINAL_RAW_DONE > /tmp/final_raw.flag
