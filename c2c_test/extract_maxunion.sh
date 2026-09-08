#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Extract the MAXUNION sweep into a flat table, COMPUTING every guard at EVERY
# point (prereg S6 G1..G11).  Nothing here is asserted from a spot check.
# Dir naming: <arm>_be<big_every>_pv<priv_lines>
W=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/maxunion/sweep
cd $W || exit 1
echo "arm BE PV parsed_knob cycles insn ctas cta_limit ctas_per_sm smem_cta_B sim_static_smem_B sim_dyn_smem_B l1_assoc_postsplit l1_KB_postsplit l1_unified_KB nbig nsmall idle_frac stages_smem holds_big l1mr l2mr resfail dram served expired mrk_hit mrk_miss mrk_hr unm_hit unm_miss unm_hr walks prot_seen biased spills spill_rate quota_ovr setfull no_epoch leg_agree leg_differ stamp_unmk exit bit ck ref"
for d in */; do
  d=${d%/}
  [ -f "$d/run.log" ] || continue
  arm=$(echo $d | cut -d_ -f1)
  BE=$(echo $d | cut -d_ -f2 | sed 's/^be//')
  PV=$(echo $d | cut -d_ -f3 | sed 's/^pv//')
  ec=$(cat $d/exit.code 2>/dev/null || echo NA)
  L=$d/run.log
  # G9: the knob the SIMULATOR parsed, not what we wrote
  pk=$(grep -a "gpgpu_cluster_barrier_invalidate_l1" $L | head -1 | awk '{print $2}')
  cyc=$(grep -a "^gpu_tot_sim_cycle" $L | tail -1 | awk '{print $NF}')
  # G10: instruction count -- a cycles gap with no insn explanation is not a
  # mechanism result until the nvcc-unrolling confound is excluded
  insn=$(grep -a "^gpu_tot_sim_insn" $L | tail -1 | awk '{print $NF}')
  ctas=$(grep -a "^gpu_tot_issued_cta" $L | tail -1 | awk '{print $NF}')
  # G2: occupancy.  CAREFUL: the simulator's "CTA/core = N" line prints at
  # shader.cc:3565, which is BEFORE the k.num_blocks() clamp at :3576-3578, so N
  # is the RESOURCE LIMIT (what would fit), not the CTAs/SM actually resident.
  # Both are recorded: cta_limit is the resource limit; ctas_per_sm is the real
  # occupancy, computed as ceil(blocks / SMs) from THIS run's own numbers.
  ctalim=$(grep -a "CTA/core = " $L | head -1 | sed -n 's/.*CTA\/core = \([0-9]*\).*/\1/p')
  nsm=$(grep -a "^-gpgpu_n_clusters" $L | head -1 | awk '{print $2}')
  cps=NA
  if [ -n "$ctas" ] && [ -n "$nsm" ] && [ "$nsm" != "0" ]; then
    cps=$(awk -v b=$ctas -v s=$nsm -v lim=${ctalim:-0} 'BEGIN{
      r=int(b/s); if (b%s) r++;              # ceil(blocks/SMs)
      if (lim>0 && r>lim) r=lim;             # never exceed the resource limit
      print r}')
  fi
  # G3 INDEPENDENT READBACK: the simulator's own launch banner, not the
  # benchmark's self-report (libcuda/cuda_runtime_api.cc:1040-1042)
  sstat=$(grep -a "Static Shared Memory Per Block" $L | head -1 | awk '{print $(NF-1)}')
  sdyn=$(grep -a "Dynamic Shared Memory Per Block" $L | head -1 | awk '{print $(NF-1)}')
  # G4 / KILL-M9: the AUTHORITATIVE post-split L1 size.  The "-gpgpu_cache:dl1"
  # echo shows the BASE assoc (64) before the adaptive split, so it must NOT be
  # used for KILL-M9.  shader.cc:3635-3636 prints the size AFTER set_assoc(), as
  # "GPGPU-Sim: Reconfigure L1 cache to <N>KB" -- that is the L1 this arm really
  # gets, and it is what addendum-01 A3 pre-registers as 28 / 224 / 256 KB.
  l1kb=$(grep -a "Reconfigure L1 cache to" $L | head -1 | sed -n 's/.*to \([0-9]*\)KB.*/\1/p')
  assoc_base=$(grep -a "^-gpgpu_cache:dl1 " $L | head -1 | sed -n 's/.*S:\([0-9]*\):\([0-9]*\):\([0-9]*\),.*/\3/p')
  nset=$(grep -a "^-gpgpu_cache:dl1 " $L | head -1 | sed -n 's/.*S:\([0-9]*\):\([0-9]*\):\([0-9]*\),.*/\1/p')
  lsz=$(grep -a "^-gpgpu_cache:dl1 " $L | head -1 | sed -n 's/.*S:\([0-9]*\):\([0-9]*\):\([0-9]*\),.*/\2/p')
  uni=$(grep -a "^-gpgpu_unified_l1d_size" $L | head -1 | awk '{print $2}')
  # derive the POST-split assoc from the reported KB (assoc = KB*1024/(nset*line))
  assoc=NA
  if [ -n "$l1kb" ] && [ -n "$nset" ] && [ -n "$lsz" ] && [ "$nset" != "0" ]; then
    assoc=$(awk -v k=$l1kb -v n=$nset -v l=$lsz 'BEGIN{printf "%.0f", k*1024/(n*l)}')
  fi
  l1mr=$(grep -a "L1D_total_cache_miss_rate" $L | tail -1 | awk '{print $NF}')
  l2mr=$(grep -a "L2_total_cache_miss_rate" $L | tail -1 | awk '{print $NF}')
  # G5 (KILL-M8)
  resfail=$(grep -a "L1D_total_cache_reservation_fails" $L | tail -1 | awk '{print $NF}')
  dram=$(grep -a "^gpgpu_n_mem_read_global" $L | tail -1 | awk '{print $NF}')
  gs=$(grep -a "RESC_GEN_STATS" $L | tail -1)
  served=$(echo "$gs" | sed -n 's/.*readshare_served=\([0-9]*\).*/\1/p')
  expired=$(echo "$gs" | sed -n 's/.*readshare_expired=\([0-9]*\).*/\1/p')
  sp=$(grep -a "RESC_L1SPLIT_STATS" $L | tail -1)
  mh=$(echo "$sp" | sed -n 's/.*[^n]marked_rd_hit=\([0-9]*\).*/\1/p')
  mm=$(echo "$sp" | sed -n 's/.*[^n]marked_rd_miss=\([0-9]*\).*/\1/p')
  mhr=$(echo "$sp" | sed -n 's/.*[^n]marked_hitrate=\([0-9.]*\).*/\1/p')
  uh=$(echo "$sp" | sed -n 's/.*unmarked_rd_hit=\([0-9]*\).*/\1/p')
  um=$(echo "$sp" | sed -n 's/.*unmarked_rd_miss=\([0-9]*\).*/\1/p')
  uhr=$(echo "$sp" | sed -n 's/.*unmarked_hitrate=\([0-9.]*\).*/\1/p')
  # G6/G7: retention self-checks and internal consistency
  rt=$(grep -a "RESC_RETAIN_STATS" $L | tail -1)
  walks=$(echo "$rt" | sed -n 's/.*victim_walks=\([0-9]*\).*/\1/p')
  pseen=$(echo "$rt" | sed -n 's/.*prot_candidates_seen=\([0-9]*\).*/\1/p')
  biased=$(echo "$rt" | sed -n 's/.*biased=\([0-9]*\).*/\1/p')
  spills=$(echo "$rt" | sed -n 's/.*spills=\([0-9]*\).*/\1/p')
  srate=$(echo "$rt" | sed -n 's/.*spill_rate=\([0-9.]*\).*/\1/p')
  qovr=$(echo "$rt" | sed -n 's/.*quota_override=\([0-9]*\).*/\1/p')
  sfull=$(echo "$rt" | sed -n 's/.*set_full=\([0-9]*\).*/\1/p')
  noep=$(echo "$rt" | sed -n 's/.*no_epoch=\([0-9]*\).*/\1/p')
  lagr=$(echo "$rt" | sed -n 's/.*legacy_agree=\([0-9]*\).*/\1/p')
  ldif=$(echo "$rt" | sed -n 's/.*legacy_differ=\([0-9]*\).*/\1/p')
  sunm=$(echo "$rt" | sed -n 's/.*stamp_unmarked=\([0-9]*\).*/\1/p')
  # G1/G3: correctness and the arm's OWN reported SMEM/CTA + idle fraction
  mu=$(grep -a "^MAXUNION arm=" $L | tail -1)
  bit=$(echo "$mu" | sed -n 's/.*BITIDENTICAL=\([0-9]*\).*/\1/p')
  ck=$(echo "$mu"  | sed -n 's/.*OUT_CHECKSUM=\([0-9]*\).*/\1/p')
  ref=$(echo "$mu" | sed -n 's/.*REF_CHECKSUM=\([0-9]*\).*/\1/p')
  smem=$(echo "$mu" | sed -n 's/.*SMEM_PER_CTA_B=\([0-9]*\).*/\1/p')
  nbig=$(echo "$mu" | sed -n 's/.*nbig=\([0-9]*\).*/\1/p')
  nsml=$(echo "$mu" | sed -n 's/.*nsmall=\([0-9]*\).*/\1/p')
  idle=$(echo "$mu" | sed -n 's/.*idle_reserved_frac=\([0-9.]*\).*/\1/p')
  stg=$(echo "$mu"  | sed -n 's/.*stages_in_smem=\([0-9]*\).*/\1/p')
  # KILL-M10: does this arm's arena hold the largest payload it may be handed?
  hb=$(echo "$mu"   | sed -n 's/.*holds_big_payload=\([0-9]*\).*/\1/p')
  echo "$arm $BE $PV ${pk:-NA} ${cyc:-NA} ${insn:-NA} ${ctas:-NA} ${ctalim:-NA} ${cps:-NA} ${smem:-NA} ${sstat:-NA} ${sdyn:-NA} ${assoc:-NA} ${l1kb:-NA} ${uni:-NA} ${nbig:-NA} ${nsml:-NA} ${idle:-NA} ${stg:-NA} ${hb:-NA} ${l1mr:-NA} ${l2mr:-NA} ${resfail:-NA} ${dram:-NA} ${served:-NA} ${expired:-NA} ${mh:-NA} ${mm:-NA} ${mhr:-NA} ${uh:-NA} ${um:-NA} ${uhr:-NA} ${walks:-NA} ${pseen:-NA} ${biased:-NA} ${spills:-NA} ${srate:-NA} ${qovr:-NA} ${sfull:-NA} ${noep:-NA} ${lagr:-NA} ${ldif:-NA} ${sunm:-NA} $ec ${bit:-NA} ${ck:-NA} ${ref:-NA}"
done | sort -k3,3n -k2,2n -k1,1
