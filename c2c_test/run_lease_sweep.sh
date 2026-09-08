#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Full pre-registered S x f sweep for the RESC epoch/version LEASE (candidate #1).
# Arms per cell: OFF(0), arm0(1), arm2(2), lease(3), leaseNK-ablation(3+nokeep).
# Records sim-INDEPENDENT traffic (injection/fabric-GET events, L1D misses, DRAM
# reads) + gpu_tot_sim_cycle + my lease counters + BITIDENTICAL, per arm.
# cluster fixed 8 (CS baked into binary). TL_L1 = L1-resident WS, TL_L2 = L2-resident.
set -u
source ~/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cp cfg_h20.config gpgpusim.config

NC=${NC:-1}          # num clusters
NE=${NE:-8}          # epochs
TO=${TO:-400}
OUT=${OUT:-LEASE_SWEEP_RESULTS.txt}
TL_L1=${TL_L1:-4}    # L1-resident working set (lines)
TL_L2=${TL_L2:-512}  # L2-resident working set (lines)

getcyc(){ echo "$1"|grep "gpu_tot_sim_cycle ="|tail -1|grep -oE "[0-9]+"; }
getev(){ echo "$1"|grep RESC_INJECT_STATS|grep -oE "events=[0-9]+"|cut -d= -f2; }
getl1m(){ echo "$1"|grep "L1D_total_cache_misses"|grep -oE "[0-9]+"; }
getdram(){ echo "$1"|grep -iE "total dram (reads|Reads)"|tail -1|grep -oE "[0-9]+"; }
getbit(){ echo "$1"|grep -oE "BITIDENTICAL=[0-9]"|cut -d= -f2; }
getref(){ echo "$1"|grep RESC_LEASE_STATS|grep -oE "refetch=[0-9]+"|head -1|cut -d= -f2; }
getkept(){ echo "$1"|grep RESC_LEASE_STATS|grep -oE "kept=[0-9]+"|cut -d= -f2; }
run(){ # mode nokeep TL f S -> echoes "cyc ev l1m dram bit refetch kept"
  local mode=$1 nk=$2 TL=$3 f=$4 S=$5
  local o; o=$(timeout $TO env RESC_LEASE=$mode RESC_LEASE_NOKEEP=$nk ./lease_bench $NC $NE $TL $f $S 2>/dev/null)
  echo "$(getcyc "$o") $(getev "$o") $(getl1m "$o") $(getdram "$o") $(getbit "$o") $(getref "$o") $(getkept "$o")"
}

{
echo "# RESC EPOCH/VERSION LEASE sweep $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "# host $(hostname); cfg_h20; CS=8 NC=$NC NE=$NE; TL_L1=$TL_L1 TL_L2=$TL_L2; inject charge ON"
echo "# lease budget = default(inf ticks): keep driven by version-certificate + workload S/f"
echo "# arms: OFF=perfect-keep  arm0=always-global  arm2=self-inval-all-each-sync  lease=keep-if-version-dominates  leaseNK=lease-disabled-ablation(==arm2)"
echo "# metric per arm: cyc=gpu_tot_sim_cycle  ev=fabric-GET(injection events)  l1m=L1D misses  dram=DRAM reads  bit=BITIDENTICAL  rf=my_refetch  kp=my_kept"
echo "# WIN rule cell: S>=2,f<=0.5 -> (ev+l1m+dram) cut >=15% AND cyc cut >=10% vs arm2, ablation==arm2"
echo "#"
} > "$OUT"

for WS in L1 L2; do
  TL=$TL_L1; [ "$WS" = L2 ] && TL=$TL_L2
  echo "===== WORKING SET = $WS (tile_lines=$TL, $((TL*128/1024))KB/cluster) =====" | tee -a "$OUT"
  printf "%-3s %-5s | %-28s | %-28s | %-28s | %-28s | %-28s\n" \
    "S" "f%%" "OFF(cyc ev l1m dram bit)" "arm0" "arm2" "lease(.. rf kp)" "leaseNK(ablation)" | tee -a "$OUT"
  for S in 1 2 4 8; do
    for f in 0 100 500 1000; do
      OFFv=$(run 0 0 $TL $f $S)
      A0v=$(run 1 0 $TL $f $S)
      A2v=$(run 2 0 $TL $f $S)
      LEv=$(run 3 0 $TL $f $S)
      NKv=$(run 3 1 $TL $f $S)
      # compute ratios lease/arm2 on traffic(ev+l1m+dram) and cycles
      read a2cyc a2ev a2l1 a2dr a2bit a2rf a2kp <<< "$A2v"
      read lecyc leev lel1 ledr lebit lerf lekp <<< "$LEv"
      trf_a2=$(( ${a2ev:-0} + ${a2l1:-0} + ${a2dr:-0} ))
      trf_le=$(( ${leev:-0} + ${lel1:-0} + ${ledr:-0} ))
      cycr=$(awk "BEGIN{if(\"$a2cyc\"!=\"\"&&$a2cyc>0)printf \"%.3f\",$lecyc/$a2cyc;else print \"NA\"}")
      trfr=$(awk "BEGIN{if($trf_a2>0)printf \"%.3f\",$trf_le/$trf_a2;else print \"NA\"}")
      printf "%-3s %-5s | %-28s | %-28s | %-28s | %-28s | %-28s | lease/arm2 cyc=%s trf=%s\n" \
        "$S" "$f" "$OFFv" "$A0v" "$A2v" "$LEv" "$NKv" "$cycr" "$trfr" | tee -a "$OUT"
    done
  done
done
echo "LEASE_SWEEP_DONE" | tee -a "$OUT"
