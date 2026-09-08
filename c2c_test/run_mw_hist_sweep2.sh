#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# run_mw_hist_sweep2.sh -- arm0 (correct-global) vs arm-dsm (single-owner funnel)
# full 9-cell regime characterization. arm3 (RESC_MW=1) DEADLOCKS at UPD>=8 (proven
# by boundary probe + author's own mw_reduce_atomic@rounds>=2), so it is run on 2
# confirmation cells only and recorded as DEADLOCK for the rest.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cp cfg_h20.config gpgpusim.config

UPD=512; NCL=8; BLK=64; TO=600
OUT=MW_CONTENTION_RESULTS.txt

getcyc(){ echo "$1"|grep "gpu_tot_sim_cycle ="|tail -1|grep -oE "[0-9]+"; }
getdram(){ echo "$1"|grep "total dram reads ="|tail -1|grep -oE "[0-9]+"; }
getbit(){ echo "$1"|grep "MW_HIST"|grep -oE "BITIDENTICAL=[0-9]+"|cut -d= -f2; }
gethot(){ echo "$1"|grep "MW_HIST"|grep -oE "frac0=[0-9.]+"|cut -d= -f2; }
getdead(){ echo "$1"|grep -c "deadlock detected"; }

{
echo "# MW skewed atomic-scatter contention sweep  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "# HOST $(uptime)"
echo "# cfg_h20 cluster=8 payload=4B(32bit atomic) UPD=$UPD/rank nclusters=$NCL block=$BLK"
echo "# arm0=correct-global  arm-dsm=single-owner SMEM funnel(map_shared_rank)"
echo "# arm3=RESC_MW=1 owner-token -> DEADLOCKS at UPD>=8 (intrinsic, see boundary probe)"
echo "# cols: K skew hot0frac cyc0 cycDSM dram0 dramDSM bit0 bitDSM dsm/glob(>1=DSM_slower)"
} > "$OUT"

for K in 32 64 256; do
 for SK in 0 1 2; do
   nm=$([ $SK -eq 0 ]&&echo uniform||([ $SK -eq 1 ]&&echo zipf1.0||echo zipf1.5))
   o0=$(timeout $TO ./hist_mw_base global $K $SK $UPD $NCL $BLK 2>/dev/null)
   od=$(timeout $TO ./hist_mw       dsm    $K $SK $UPD $NCL $BLK 2>/dev/null)
   c0=$(getcyc "$o0"); cd=$(getcyc "$od")
   d0=$(getdram "$o0"); dd=$(getdram "$od")
   b0=$(getbit "$o0"); bd=$(getbit "$od"); hf=$(gethot "$o0")
   rr=$(awk "BEGIN{if(\"$c0\"!=\"\"&&$c0>0&&\"$cd\"!=\"\")printf \"%.3f\",$cd/$c0; else print \"NA\"}")
   echo "K=$K skew=$nm hot0=$hf cyc0=$c0 cycDSM=$cd dram0=$d0 dramDSM=$dd bit=$b0/$bd dsm_over_glob=${rr}x" | tee -a "$OUT"
 done
done

echo "# --- arm3 RESC_MW=1 deadlock confirmation at sweep UPD=$UPD (2 cells) ---" | tee -a "$OUT"
for spec in "64 2" "256 0"; do
  set -- $spec; K=$1; SK=$2
  nm=$([ $SK -eq 0 ]&&echo uniform||([ $SK -eq 1 ]&&echo zipf1.0||echo zipf1.5))
  o3=$(RESC_MW=1 timeout 250 ./hist_mw global $K $SK $UPD $NCL $BLK 2>&1)
  ec=$?
  echo "arm3 K=$K skew=$nm UPD=$UPD exit=$ec deadlock=$(getdead "$o3") cyc=$(getcyc "$o3" | head -1)" | tee -a "$OUT"
done
echo "MW_HIST_SWEEP_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT"
