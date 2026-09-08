#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
set -u
source ~/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cp cfg_h20.config gpgpusim.config
NC=1; NE=8; TO=400; TL=4; OUT=LEASE_SWEEP_L1.txt
c(){ echo "$1"|grep "gpu_tot_sim_cycle ="|tail -1|grep -oE "[0-9]+"; }
ev(){ echo "$1"|grep RESC_INJECT_STATS|grep -oE "events=[0-9]+"|cut -d= -f2; }
lm(){ echo "$1"|grep "L1D_total_cache_misses"|grep -oE "[0-9]+"|head -1; }
dr(){ echo "$1"|grep -iE "total dram (reads|Reads)"|tail -1|grep -oE "[0-9]+"; }
bt(){ echo "$1"|grep -oE "BITIDENTICAL=[0-9]"|cut -d= -f2; }
rf(){ echo "$1"|grep RESC_LEASE_STATS|grep -oE "refetch=[0-9]+"|head -1|cut -d= -f2; }
kp(){ echo "$1"|grep RESC_LEASE_STATS|grep -oE "kept=[0-9]+"|cut -d= -f2; }
run(){ local o; o=$(timeout $TO env RESC_LEASE=$1 RESC_LEASE_NOKEEP=$2 ./lease_bench $NC $NE $TL $3 $4 2>/dev/null); echo "cyc=$(c "$o") ev=$(ev "$o") l1m=$(lm "$o") dram=$(dr "$o") bit=$(bt "$o") rf=$(rf "$o") kp=$(kp "$o")"; }
{
echo "# LEASE L1-resident sweep (TL=$TL, 0.5KB/consumer) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "# cols per arm: cyc ev(fabric-GET) l1m dram bit rf(myrefetch) kp(mykept)"
} > "$OUT"
for S in 1 2 4 8; do for f in 0 100 500 1000; do
  echo "--- S=$S f_permille=$f ---" >> "$OUT"
  echo "  OFF    : $(run 0 0 $f $S)" >> "$OUT"
  echo "  arm0   : $(run 1 0 $f $S)" >> "$OUT"
  A2=$(run 2 0 $f $S); echo "  arm2   : $A2" >> "$OUT"
  LE=$(run 3 0 $f $S); echo "  lease  : $LE" >> "$OUT"
  NK=$(run 3 1 $f $S); echo "  leaseNK: $NK" >> "$OUT"
  a2c=$(echo $A2|grep -oE "cyc=[0-9]+"|cut -d= -f2); a2e=$(echo $A2|grep -oE "ev=[0-9]+"|cut -d= -f2); a2l=$(echo $A2|grep -oE "l1m=[0-9]+"|cut -d= -f2); a2d=$(echo $A2|grep -oE "dram=[0-9]+"|cut -d= -f2)
  lec=$(echo $LE|grep -oE "cyc=[0-9]+"|cut -d= -f2); lee=$(echo $LE|grep -oE "ev=[0-9]+"|cut -d= -f2); lel=$(echo $LE|grep -oE "l1m=[0-9]+"|cut -d= -f2); led=$(echo $LE|grep -oE "dram=[0-9]+"|cut -d= -f2)
  ta=$((${a2e:-0}+${a2l:-0}+${a2d:-0})); tl=$((${lee:-0}+${lel:-0}+${led:-0}))
  echo "  RATIO lease/arm2: cyc=$(awk "BEGIN{if(${a2c:-0}>0)printf \"%.3f\",${lec:-0}/${a2c};else print \"NA\"}") traffic=$(awk "BEGIN{if($ta>0)printf \"%.3f\",$tl/$ta;else print \"NA\"}") (a2_trf=$ta le_trf=$tl)" >> "$OUT"
done; done
echo "LEASE_L1_DONE" >> "$OUT"
