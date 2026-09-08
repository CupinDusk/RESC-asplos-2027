#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source ~/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cp cfg_h20.config gpgpusim.config
OUT=DPATH_N8_DECISIVE.txt
TO=500
getc(){ echo "$1"|grep "gpu_tot_sim_cycle ="|tail -1|grep -oE "[0-9]+"; }
getf(){ echo "$1"|grep "DCOMB "; }
getdram(){ echo "$1"|grep "total dram reads ="|tail -1|grep -oE "[0-9]+"; }
echo "# N=8 DECISIVE cells (defensible regime) $(date -u +%Y-%m-%dT%H:%M:%SZ)" > $OUT
echo "# COLS: T K LOCAL | tot g/f/t | comb g/f/t | tree/f tree/g | dram g/f/t | bit g/f/t | round_cyc" >> $OUT
for T in 1 0; do
 for K in 32 256; do
  for L in 16 64; do
   G=$(timeout $TO ./dcomb_n8_t${T} global $K $L 8 32 2>/dev/null)
   F=$(timeout $TO ./dcomb_n8_t${T} funnel $K $L 8 32 2>/dev/null)
   R=$(timeout $TO ./dcomb_n8_t${T} tree   $K $L 8 32 2>/dev/null)
   tg=$(getc "$G"); tf=$(getc "$F"); tr=$(getc "$R")
   cg=$(getf "$G"|grep -oE "rank0_comb=[0-9]+"|cut -d= -f2)
   cf=$(getf "$F"|grep -oE "rank0_comb=[0-9]+"|cut -d= -f2)
   ct=$(getf "$R"|grep -oE "rank0_comb=[0-9]+"|cut -d= -f2)
   dg=$(getdram "$G"); df=$(getdram "$F"); dr=$(getdram "$R")
   bg=$(getf "$G"|grep -oE "BITIDENTICAL=[0-9]"|cut -d= -f2)
   bf=$(getf "$F"|grep -oE "BITIDENTICAL=[0-9]"|cut -d= -f2)
   br=$(getf "$R"|grep -oE "BITIDENTICAL=[0-9]"|cut -d= -f2)
   rc=$(getf "$R"|grep -oE "round_cyc=\[[0-9,]*\]")
   rtf=$(awk "BEGIN{if(\"$tf\"!=\"\"&&$tf>0)printf \"%.3f\",$tr/$tf; else print \"NA\"}")
   rtg=$(awk "BEGIN{if(\"$tg\"!=\"\"&&$tg>0)printf \"%.3f\",$tr/$tg; else print \"NA\"}")
   echo "T=$T K=$K L=$L | tot g=$tg f=$tf t=$tr | comb g=$cg f=$cf t=$ct | tree/f=$rtf tree/g=$rtg | dram g=$dg f=$df t=$dr | bit=$bg/$bf/$br | $rc" >> $OUT
  done
 done
done
echo "N8_DECISIVE_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $OUT
