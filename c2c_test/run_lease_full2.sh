#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Sequential S x f sweep for the RESC epoch/version LEASE. No conda activate
# (LD_LIBRARY_PATH/PATH set directly). Each regime runs in its OWN dir with its
# OWN gpgpusim.config so concurrent/other runs cannot clobber it.
BASE=/legacy-research/owner/cudatest/RESC-c2c-wt
export PATH=$BASE/c2c_test/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=$BASE/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
OUT=$BASE/c2c_test/LEASE_SWEEP_FULL.txt

c(){ echo "$1"|grep "gpu_tot_sim_cycle ="|tail -1|grep -oE "[0-9]+"; }
evf(){ echo "$1"|grep RESC_INJECT_STATS|grep -oE "events=[0-9]+"|cut -d= -f2; }
lm(){ echo "$1"|grep "L1D_total_cache_misses"|grep -oE "[0-9]+"|head -1; }
dr(){ echo "$1"|grep -iE "total dram (reads|Reads)"|tail -1|grep -oE "[0-9]+"; }
bt(){ echo "$1"|grep -oE "BITIDENTICAL=[0-9]"|cut -d= -f2; }
rf(){ echo "$1"|grep RESC_LEASE_STATS|grep -oE "refetch=[0-9]+"|head -1|cut -d= -f2; }
kp(){ echo "$1"|grep RESC_LEASE_STATS|grep -oE "kept=[0-9]+"|cut -d= -f2; }
DIR=""; TL=4; NE=8; TO=400
runarm(){ # mode nokeep f S
  local o; o=$(cd "$DIR" && timeout $TO env RESC_LEASE=$1 RESC_LEASE_NOKEEP=$2 ./lease_bench 1 $NE $TL $3 $4 2>/dev/null)
  echo "cyc=$(c "$o") ev=$(evf "$o") l1m=$(lm "$o") dram=$(dr "$o") bit=$(bt "$o") rf=$(rf "$o") kp=$(kp "$o")"
}

echo "# RESC EPOCH/VERSION LEASE full sweep $(date -u +%Y-%m-%dT%H:%M:%SZ) host $(hostname)" > "$OUT"
echo "# arms: OFF=perfect-keep arm0=always-global arm2=self-inval-all-each-sync lease=keep-if-version-dominates leaseNK=ablation(==arm2)" >> "$OUT"
echo "# WIN cell (pre-reg): S>=2,f<=0.5 -> (ev+l1m+dram) cut>=15% AND cyc cut>=10% vs arm2, ablation==arm2, bit=1" >> "$OUT"

do_regime(){ # dir cfg TL NE TO label
  DIR=$1; local cfg=$2; TL=$3; NE=$4; TO=$5; local label=$6
  (cd "$DIR" && cp "$cfg" gpgpusim.config)
  echo "===== REGIME=$label dir=$DIR cfg=$cfg TL=$TL ($((TL*128/1024))KB/consumer) NE=$NE =====" >> "$OUT"
  echo "# cols per arm: cyc ev(fabric-GET) l1m(L1D-miss) dram bit rf(myrefetch) kp(mykept)" >> "$OUT"
  local S f A2 LE NK
  for S in 1 2 4 8; do for f in 0 100 500 1000; do
    echo "--- S=$S f_permille=$f ---" >> "$OUT"
    echo "  OFF    : $(runarm 0 0 $f $S)" >> "$OUT"
    echo "  arm0   : $(runarm 1 0 $f $S)" >> "$OUT"
    A2=$(runarm 2 0 $f $S); echo "  arm2   : $A2" >> "$OUT"
    LE=$(runarm 3 0 $f $S); echo "  lease  : $LE" >> "$OUT"
    NK=$(runarm 3 1 $f $S); echo "  leaseNK: $NK" >> "$OUT"
    local a2c a2e a2l a2d lec lee lel led ta tl
    a2c=$(echo $A2|grep -oE "cyc=[0-9]+"|cut -d= -f2); a2e=$(echo $A2|grep -oE "ev=[0-9]+"|cut -d= -f2); a2l=$(echo $A2|grep -oE "l1m=[0-9]+"|cut -d= -f2); a2d=$(echo $A2|grep -oE "dram=[0-9]+"|cut -d= -f2)
    lec=$(echo $LE|grep -oE "cyc=[0-9]+"|cut -d= -f2); lee=$(echo $LE|grep -oE "ev=[0-9]+"|cut -d= -f2); lel=$(echo $LE|grep -oE "l1m=[0-9]+"|cut -d= -f2); led=$(echo $LE|grep -oE "dram=[0-9]+"|cut -d= -f2)
    ta=$(( ${a2e:-0}+${a2l:-0}+${a2d:-0} )); tl=$(( ${lee:-0}+${lel:-0}+${led:-0} ))
    echo "  RATIO lease/arm2: cyc=$(awk "BEGIN{if(${a2c:-0}>0)printf \"%.3f\",${lec:-0}/${a2c};else print \"NA\"}") traffic=$(awk "BEGIN{if($ta>0)printf \"%.3f\",$tl/$ta;else print \"NA\"}") (a2_trf=$ta le_trf=$tl)" >> "$OUT"
  done; done
}

do_regime $BASE/c2c_test/lease_wd_l1 cfg_h20.config          4   8 400 "L1-RESIDENT"
do_regime $BASE/c2c_test/lease_wd_l2 cfg_h20_smallL1.config 256 3 590 "L2-RESIDENT-smallL1-16KB"
echo "LEASE_SWEEP_FULL_DONE" >> "$OUT"
