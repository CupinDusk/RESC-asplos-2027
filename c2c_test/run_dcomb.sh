#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source ~/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cp cfg_h20.config gpgpusim.config
TO=400; NCL=8; BLK=32
OUT=DPATH_COMBINING_RESULTS.txt
getcyc(){ echo "$1"|grep "gpu_tot_sim_cycle ="|tail -1|grep -oE "[0-9]+"; }
getdram(){ echo "$1"|grep "total dram reads ="|tail -1|grep -oE "[0-9]+"; }
getf(){ echo "$1"|grep "DCOMB "; }
{
echo "# D-PATH reducible epoch combining sweep $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "# HOST: $(uptime)"
echo "# cfg_h20 n_cores_per_cluster=1 sm_2_sm_network_latency=195; nclust=$NCL block=$BLK"
echo "# arms: global=atomic-shared-bins  funnel=owner-serial(N-1)  tree=reduction-tree(log2 N)"
echo "# metric: gpu_tot_sim_cycle (wall-clock) per arm; rank0_comb = critical-path combine cyc"
echo "# COLS: N T K LOCAL | tot_global tot_funnel tot_tree | comb_g comb_f comb_t | tree/funnel tree/global | dram_t | bit | round_cyc"
} > "$OUT"

for N in 4 8; do
 for T in 1 0; do
  for K in 1 32 256; do
   for LOCAL in 16 64 256; do
     G=$(timeout $TO ./dcomb_n${N}_t${T} global $K $LOCAL $NCL $BLK 2>/dev/null)
     F=$(timeout $TO ./dcomb_n${N}_t${T} funnel $K $LOCAL $NCL $BLK 2>/dev/null)
     R=$(timeout $TO ./dcomb_n${N}_t${T} tree   $K $LOCAL $NCL $BLK 2>/dev/null)
     tg=$(getcyc "$G"); tf=$(getcyc "$F"); tr=$(getcyc "$R")
     cg=$(getf "$G"|grep -oE "rank0_comb=[0-9]+"|cut -d= -f2)
     cf=$(getf "$F"|grep -oE "rank0_comb=[0-9]+"|cut -d= -f2)
     ct=$(getf "$R"|grep -oE "rank0_comb=[0-9]+"|cut -d= -f2)
     dt=$(getdram "$R")
     bg=$(getf "$G"|grep -oE "BITIDENTICAL=[0-9]"|cut -d= -f2)
     bf=$(getf "$F"|grep -oE "BITIDENTICAL=[0-9]"|cut -d= -f2)
     br=$(getf "$R"|grep -oE "BITIDENTICAL=[0-9]"|cut -d= -f2)
     rc=$(getf "$R"|grep -oE "round_cyc=\[[0-9,]*\]")
     rtf=$(awk "BEGIN{if(\"$tf\"!=\"\"&&$tf>0&&\"$tr\"!=\"\")printf \"%.3f\",$tr/$tf; else print \"NA\"}")
     rtg=$(awk "BEGIN{if(\"$tg\"!=\"\"&&$tg>0&&\"$tr\"!=\"\")printf \"%.3f\",$tr/$tg; else print \"NA\"}")
     echo "N=$N T=$T K=$K LOCAL=$LOCAL | tot g=$tg f=$tf t=$tr | comb g=$cg f=$cf t=$ct | tree/f=${rtf} tree/g=${rtg} | dram_t=$dt | bit=$bg/$bf/$br | $rc" | tee -a "$OUT"
   done
  done
 done
done
echo "DCOMB_SWEEP_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT"
