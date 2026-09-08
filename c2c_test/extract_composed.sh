#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Extract the COMPOSED-REGIME sweep, COMPUTING every mandatory column at every
# point.  Prereg P6: any cycle claim without SMEM/CTA and the ACTUAL CTAs/SM is
# void.  Nothing here is asserted from a spot check.
W=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/composed/sweep
cd $W || exit 1
printf "%s\n" "point arm arm_name exp_knob parsed_knob knob_guard cycles insn ctas cta_limit ctas_per_sm p6_guard smem_cta_B sim_static_smem_B sim_dyn_smem_B l1_KB l1mr priv_unm_hit priv_unm_miss priv_unm_hr mrk_hit mrk_miss mrk_hr whole_inv sel_inv lines_exam mrk_invd unm_presv served expired dram resfail bit ck ref exit"
for d in */; do
  d=${d%/}
  [ -f "$d/run.log" ] || continue
  case "$d" in CONTAMINATED*|INVALID*|*_INVALID*) continue;; esac
  # OWNERSHIP FILTER: report only the arms this experiment defines (0,2,5,6,7).
  # A concurrent, separately-preregistered experiment (run_nofwd.sh, RESC_NO_FWD=1)
  # writes *_A8 cells into this same sweep directory; those are not ours.
  case "$d" in *_A0|*_A2|*_A5|*_A6|*_A7) ;; *) continue;; esac
  pn=${d%_*}; arm=${d##*_}
  ec=$(cat $d/exit.code 2>/dev/null || echo NA)
  L=$d/run.log
  mu=$(grep -a "^MAXUNION arm=" $L | tail -1)
  gv(){ echo "$mu" | grep -oE "$1=[0-9.]+" | head -1 | cut -d= -f2; }
  aname=$(echo "$mu" | grep -oE "arm_name=[a-zA-Z_]+" | cut -d= -f2)
  ek=$(gv expected_knob)
  # G9: the knob the SIMULATOR parsed, not what we wrote -- evaluated, not asserted
  pk=$(grep -a "gpgpu_cluster_barrier_invalidate_l1" $L | head -1 | awk '{print $2}')
  kg=NA; [ -n "$ek" ] && [ -n "$pk" ] && { [ "$ek" = "$pk" ] && kg=PASS || kg=FAIL; }
  cyc=$(grep -a "^gpu_tot_sim_cycle" $L | tail -1 | awk '{print $NF}')
  insn=$(grep -a "^gpu_tot_sim_insn" $L | tail -1 | awk '{print $NF}')
  ctas=$(grep -a "^gpu_tot_issued_cta" $L | tail -1 | awk '{print $NF}')
  # P6: shader.cc:3565 prints the RESOURCE LIMIT (pre-clamp); the real occupancy
  # is ceil(blocks/SMs) capped by that limit (post-clamp, :3576).
  ctalim=$(grep -a "CTA/core = " $L | head -1 | sed -n 's/.*CTA\/core = \([0-9]*\).*/\1/p')
  nsm=$(grep -a "^-gpgpu_n_clusters" $L | head -1 | awk '{print $2}')
  cps=NA
  if [ -n "$ctas" ] && [ -n "$nsm" ] && [ "$nsm" != "0" ]; then
    cps=$(awk -v b=$ctas -v s=$nsm -v lim=${ctalim:-0} 'BEGIN{r=int(b/s); if(b%s)r++;
      if(lim>0&&r>lim)r=lim; print r}')
  fi
  sdyn=$(grep -a "Dynamic Shared Memory Per Block" $L | head -1 | awk '{print $(NF-1)}')
  sstat=$(grep -a "Static Shared Memory Per Block" $L | head -1 | awk '{print $(NF-1)}')
  l1kb=$(grep -a "Reconfigure L1 cache to" $L | head -1 | sed -n 's/.*to \([0-9]*\)KB.*/\1/p')
  l1mr=$(grep -a "L1D_total_cache_miss_rate" $L | tail -1 | awk '{print $NF}')
  resfail=$(grep -a "L1D_total_cache_reservation_fails" $L | tail -1 | awk '{print $NF}')
  dram=$(grep -a "^gpgpu_n_mem_read_global" $L | tail -1 | awk '{print $NF}')
  # PRIVATE-READ HIT RATE: the private working set is UNMARKED in every arm, so
  # the unmarked counters are the private-read hit rate P3 needs.
  sp=$(grep -a "RESC_L1SPLIT_STATS" $L | tail -1)
  g2(){ echo "$sp" | grep -oE "$1=[0-9.]+" | head -1 | cut -d= -f2; }
  uh=$(g2 unmarked_rd_hit); um=$(g2 unmarked_rd_miss); uhr=$(g2 unmarked_hitrate)
  mh=$(echo "$sp" | grep -oE "[^n]marked_rd_hit=[0-9]+" | head -1 | cut -d= -f2)
  mm=$(echo "$sp" | grep -oE "[^n]marked_rd_miss=[0-9]+" | head -1 | cut -d= -f2)
  mhr=$(echo "$sp" | grep -oE "[^n]marked_hitrate=[0-9.]+" | head -1 | cut -d= -f2)
  si=$(grep -a "RESC_SELINV_STATS" $L | tail -1)
  g3(){ echo "$si" | grep -oE "$1=[0-9]+" | head -1 | cut -d= -f2; }
  winv=$(g3 whole_invalidates); sinv=$(g3 selective_invalidates)
  lex=$(g3 lines_examined); mi=$(g3 marked_invalidated); up=$(g3 unmarked_preserved)
  gs=$(grep -a "RESC_GEN_STATS" $L | tail -1)
  served=$(echo "$gs" | grep -oE "readshare_served=[0-9]+" | cut -d= -f2)
  expired=$(echo "$gs" | grep -oE "readshare_expired=[0-9]+" | cut -d= -f2)
  bit=$(gv BITIDENTICAL); ck=$(gv OUT_CHECKSUM); ref=$(gv REF_CHECKSUM)
  smem=$(gv SMEM_PER_CTA_B)
  # P6 (prereg): a cycle claim is VOID unless SMEM/CTA and the ACTUAL post-clamp
  # CTAs/SM are both present. Evaluated per cell, never assumed.
  p6=VOID
  if [ -n "$cps" ] && [ "$cps" != "NA" ] && [ -n "$smem" ] && [ "$smem" != "NA" ] \
     && [ -n "$sdyn" ] && [ -n "$sstat" ]; then p6=OK; fi
  echo "$pn $arm ${aname:-NA} ${ek:-NA} ${pk:-NA} $kg ${cyc:-NA} ${insn:-NA} ${ctas:-NA} ${ctalim:-NA} ${cps:-NA} $p6 ${smem:-NA} ${sstat:-NA} ${sdyn:-NA} ${l1kb:-NA} ${l1mr:-NA} ${uh:-NA} ${um:-NA} ${uhr:-NA} ${mh:-NA} ${mm:-NA} ${mhr:-NA} ${winv:-NA} ${sinv:-NA} ${lex:-NA} ${mi:-NA} ${up:-NA} ${served:-NA} ${expired:-NA} ${dram:-NA} ${resfail:-NA} ${bit:-NA} ${ck:-NA} ${ref:-NA} $ec"
done | sort -k1,1 -k2,2
