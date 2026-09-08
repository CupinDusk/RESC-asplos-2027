#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# run_mw_hist_sweep.sh -- pre-registered MW skewed-atomic-scatter contention sweep.
# arms: arm0=hist_mw_base global (correct-global), arm3=hist_mw global RESC_MW=1,
#       arm-dsm=hist_mw dsm (single-owner map_shared_rank funnel).
# sweep: K{32,64,256} x skew{0=uniform,1=zipf1.0,2=zipf1.5}, cluster=8, payload=4B.
# per cell: gpu_tot_sim_cycle per arm, ratios, BITIDENTICAL, dram reads, inject/MW stats.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cp cfg_h20.config gpgpusim.config

UPD=${UPD:-512}
NCL=${NCL:-8}
BLK=${BLK:-64}
TO=${TO:-600}
OUT=${OUT:-MW_CONTENTION_RESULTS.txt}

getcyc(){ echo "$1" | grep "gpu_tot_sim_cycle =" | tail -1 | grep -oE "[0-9]+"; }
getdram(){ echo "$1" | grep "total dram reads =" | tail -1 | grep -oE "[0-9]+"; }
getbit(){ echo "$1" | grep "MW_HIST" | grep -oE "BITIDENTICAL=[0-9]+" | cut -d= -f2; }
gethot(){ echo "$1" | grep "MW_HIST" | grep -oE "frac0=[0-9.]+" | cut -d= -f2; }
getmwmsg(){ echo "$1" | grep "RESC_MW_STATS" | tail -1 | grep -oE "new_fabric_msgs=[0-9]+" | cut -d= -f2; }
getmwtok(){ echo "$1" | grep "RESC_MW_STATS" | tail -1 | grep -oE "atomic_upgradereq=[0-9]+" | cut -d= -f2; }
getinjev(){ echo "$1" | grep "RESC_INJECT_STATS" | tail -1 | grep -oE "events=[0-9]+" | cut -d= -f2; }
getmwline(){ echo "$1" | grep "RESC_MW_STATS" | tail -1; }

{
echo "# MW skewed atomic-scatter contention sweep  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "# cfg_h20, cluster=8, payload=4B(32-bit atomic), UPD=$UPD/rank, nclusters=$NCL, block=$BLK"
echo "# arm0=correct-global  arm3=RESC_MW=1 global(owner-token)  arm-dsm=single-owner SMEM funnel"
echo "# K skew hot_frac0 cyc_arm0 cyc_dsm cyc_arm3 dram0 dramDSM dram3 bit0 bitDSM bit3 arm3/dsm arm3/arm0 mw_fabric_msgs mw_tokens inj_events(dsm)"
} > "$OUT"

for K in 32 64 256; do
 for SK in 0 1 2; do
   skname=$([ $SK -eq 0 ] && echo uniform || ([ $SK -eq 1 ] && echo zipf1.0 || echo zipf1.5))
   o0=$(timeout $TO ./hist_mw_base global $K $SK $UPD $NCL $BLK 2>/dev/null)
   od=$(timeout $TO ./hist_mw       dsm    $K $SK $UPD $NCL $BLK 2>/dev/null)
   o3=$(timeout $TO env RESC_MW=1 ./hist_mw global $K $SK $UPD $NCL $BLK 2>/dev/null)
   c0=$(getcyc "$o0"); cd=$(getcyc "$od"); c3=$(getcyc "$o3")
   d0=$(getdram "$o0"); dd=$(getdram "$od"); d3=$(getdram "$o3")
   b0=$(getbit "$o0"); bd=$(getbit "$od"); b3=$(getbit "$o3")
   hf=$(gethot "$o0")
   msg=$(getmwmsg "$o3"); tok=$(getmwtok "$o3"); iev=$(getinjev "$od")
   r3d=$(awk "BEGIN{if(\"$cd\"!=\"\" && $cd>0)printf \"%.3f\",$cd/$c3; else print \"NA\"}")
   r30=$(awk "BEGIN{if(\"$c0\"!=\"\" && $c0>0)printf \"%.3f\",$c0/$c3; else print \"NA\"}")
   echo "K=$K skew=$skname hot=$hf a0=$c0 dsm=$cd a3=$c3 dram=$d0/$dd/$d3 bit=$b0/$bd/$b3 arm3_beats_dsm=${r3d}x arm3_beats_a0=${r30}x mwmsg=$msg tok=$tok injev_dsm=$iev" | tee -a "$OUT"
   echo "    DSM_MWSTATS: $(getmwline "$od")" >> "$OUT"
   echo "    A3_MWSTATS:  $(getmwline "$o3")" >> "$OUT"
 done
done
echo "MW_HIST_SWEEP_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT"
