#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# CCOP preregistered arms: A(baseline) S1(CCOP default) S2(probe-first) C(cold-only control).
# Usage: run_ccop_arms.sh <nreg> <rfloats> <rreuse> <ncons> <tag>
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
NREG=${1:-32}; RF=${2:-512}; RR=${3:-64}; NC=${4:-4}; TAG=${5:-run}
BASE=c2c_run/$TAG
run_arm () {
  local arm=$1 mode=$2 noprobe=$3
  local d=$BASE/$arm; mkdir -p $d
  cp cfg_c2c.config $d/gpgpusim.config
  ( cd $d; RESC_C2C_MODE=$mode RESC_C2C_NOPROBE=$noprobe RESC_C2C_L2=242 \
      timeout 1800 ../../../resc_c2c $NREG $RF $RR $NC > run.log 2>&1 )
}
run_arm A  0 0
run_arm S1 1 0
run_arm S2 2 0
run_arm C  1 1
echo "# arm cycles checksum crossfrac ccop_probes fresh_hits mispredict cold_learn poison probe_bytes l2_saved strip_KB"
for arm in A S1 S2 C; do
  d=$BASE/$arm
  cyc=$(grep "gpu_tot_sim_cycle =" $d/run.log | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
  ck=$(grep OUT_CHECKSUM $d/run.log | grep -oE "sum=[0-9]+" | head -1 | sed "s/sum=//")
  kb=$(grep OUT_CHECKSUM $d/run.log | grep -oE "strip_KB=[0-9]+" | sed "s/strip_KB=//")
  xf=$(grep "\[CCOP_STATS\]" $d/run.log | tail -1 | grep -oE "crossgpc_read_frac=[0-9.]+" | sed "s/.*=//")
  pr=$(grep "\[CCOP_STATS\]" $d/run.log | tail -1 | grep -oE "ccop_probes=[0-9]+" | sed "s/.*=//")
  fh=$(grep "\[CCOP_STATS\]" $d/run.log | tail -1 | grep -oE "ccop_fresh_hits=[0-9]+" | sed "s/.*=//")
  mp=$(grep "\[CCOP_STATS\]" $d/run.log | tail -1 | grep -oE "ccop_mispredict=[0-9]+" | sed "s/.*=//")
  cl=$(grep "\[CCOP_STATS\]" $d/run.log | tail -1 | grep -oE "ccop_cold_l2_learn=[0-9]+" | sed "s/.*=//")
  po=$(grep "\[CCOP_STATS\]" $d/run.log | tail -1 | grep -oE "ccop_poison_skips=[0-9]+" | sed "s/.*=//")
  pb=$(grep "\[CCOP_STATS\]" $d/run.log | tail -1 | grep -oE "c2c_probe_bytes=[0-9]+" | sed "s/.*=//")
  ls=$(grep "\[CCOP_STATS\]" $d/run.log | tail -1 | grep -oE "l2_accesses_saved=[0-9]+" | sed "s/.*=//")
  echo "$arm ${cyc:-NA} ${ck:-NA} ${xf:-NA} ${pr:-0} ${fh:-0} ${mp:-0} ${cl:-0} ${po:-0} ${pb:-0} ${ls:-0} ${kb:-NA}"
done
echo CCOP_ARMS_DONE
